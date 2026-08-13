"""
server.py  --  lcl_pricer.py 를 웹앱이 부를 수 있게 감싸는 API
=============================================================================
사용법
    pip3 install fastapi uvicorn numpy
    python3 server.py                 # http://localhost:8000
    (문서 확인: http://localhost:8000/docs)

lcl_pricer.py / lcl_core.py 는 한 줄도 건드리지 않습니다.
이 파일을 FLEX0811 폴더 안에 두고 실행하면 됩니다.
=============================================================================
"""
from __future__ import annotations

import os
import re
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from lcl_pricer import BookingEngine, PolicyBank

BANK_PATH = os.environ.get('POLICY_BANK', 'results/policy_bank.json')
bank = PolicyBank(BANK_PATH)

app = FastAPI(title='LCL Dynamic Pricing API')
app.add_middleware(                      # ← 이거 없으면 브라우저에서 호출이 막힙니다
    CORSMiddleware,
    allow_origins=['*'], allow_methods=['*'], allow_headers=['*'],
)

# 서버가 들고 있는 예약 세션 (컨테이너 적재 상태가 누적됨)
SESSIONS: dict = {}
QUOTES: dict = {}                        # quote_id -> 원본 quote (_slots 포함)


# ---------------------------------------------------------------- 기본값
def available_seasons():
    """policy_bank 키에서 실제 학습된 시즌 조합만 뽑아냅니다."""
    out = set()
    for k in bank.keys():
        out.add(k.rsplit('|', 1)[-1])
    return sorted(out)


DEFAULTS = dict(beta=0.6, delta_F=-0.3, season='Normal-OffPeak', r_S=bank.r_S)


def season_arg(s: Optional[str]):
    if not s or s == 'ANY':
        return None
    return tuple(s.split('-')) if '-' in s else s


def new_engine(beta, delta_F, season, r_S):
    return BookingEngine(bank, beta=beta, delta_F=delta_F,
                         season=season_arg(season), r_S=r_S)


def get_engine(session_id: Optional[str]):
    sid = session_id or 'default'
    if sid not in SESSIONS:
        SESSIONS[sid] = new_engine(**DEFAULTS)
    return sid, SESSIONS[sid]


# ---------------------------------------------------------------- 스키마
class ResetIn(BaseModel):
    session_id: Optional[str] = None
    beta: float = DEFAULTS['beta']
    delta_F: float = DEFAULTS['delta_F']
    season: str = DEFAULTS['season']
    r_S: float = DEFAULTS['r_S']


class QuoteIn(BaseModel):
    length_cm: float = Field(..., gt=0, description='가로 (cm)')
    width_cm: float = Field(..., gt=0, description='세로 (cm)')
    height_cm: float = Field(..., gt=0, description='높이 (cm)')
    weight_kg: Optional[float] = Field(None, ge=0, description='중량 (kg)')
    segment: int = Field(1, ge=1, le=2, description='희망 선적 1 또는 2')
    days_to_cutoff: Optional[float] = Field(
        None, description='마감까지 남은 일수. 없으면 t 로 판단')
    t: Optional[float] = Field(None, description='판매기간 내 위치 0..T')
    session_id: Optional[str] = None


class AcceptIn(BaseModel):
    quote_id: str
    product: str = Field(..., pattern='^[SF]$')


# ---------------------------------------------------------------- 엔드포인트
@app.get('/meta')
def meta():
    """프론트가 슬라이더 범위·기본값을 잡을 때 부릅니다."""
    return {
        'axes': {'beta': bank.betas, 'delta_F': bank.dFs},
        'seasons': available_seasons(),
        'defaults': DEFAULTS,
        'horizon_days': bank.T,
        'container_cbm': bank.cbm,
        'bins': {'t_bins': bank.T_BINS, 'cap_bins': bank.CAP_BINS},
        'discounts': [a.get('d', 0.0) for a in bank.actions],
    }


@app.post('/session/reset')
def reset(inp: ResetIn):
    """운영 파라미터를 바꾸거나, 데모 촬영 전 컨테이너를 비울 때."""
    sid = inp.session_id or 'default'
    SESSIONS[sid] = new_engine(inp.beta, inp.delta_F, inp.season, inp.r_S)
    return {'session_id': sid, 'ok': True,
            'policy_key': SESSIONS[sid].policy.key,
            **SESSIONS[sid].summary()}


@app.post('/quote')
def quote(inp: QuoteIn):
    sid, eng = get_engine(inp.session_id)

    # 판매시점: 남은 일수로 받으면 t 로 환산
    if inp.t is not None:
        t = inp.t
    elif inp.days_to_cutoff is not None:
        t = eng.T - inp.days_to_cutoff
    else:
        t = eng.T * 0.5
    t = max(0.0, min(eng.T, float(t)))

    # W/M 룰: 부피와 중량 중 큰 쪽으로 청구
    vol_cbm = inp.length_cm * inp.width_cm * inp.height_cm / 1_000_000
    billed = vol_cbm
    if inp.weight_kg:
        billed = max(vol_cbm, inp.weight_kg / 1000.0)

    try:
        q = eng.quote((inp.length_cm, inp.width_cm, inp.height_cm),
                      segment=inp.segment, t=t, unit='cm', billed_cbm=billed)
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(400, f'quote 실패: {e}')

    qid = uuid.uuid4().hex[:12]
    QUOTES[qid] = (sid, q)

    out = {k: v for k, v in q.items() if k != '_slots'}      # _slots 는 서버 보관
    out['quote_id'] = qid
    out['session_id'] = sid
    out['billing'] = {'volume_cbm': round(vol_cbm, 4),
                      'weight_ton': round((inp.weight_kg or 0) / 1000.0, 4),
                      'billed_cbm': round(billed, 4),
                      'basis': 'weight' if billed > vol_cbm + 1e-9 else 'volume'}
    out['timing'] = {'t': round(t, 2), 'horizon': eng.T,
                     'days_to_cutoff': round(eng.T - t, 2)}
    return out


@app.post('/accept')
def accept(inp: AcceptIn):
    if inp.quote_id not in QUOTES:
        raise HTTPException(404, '만료된 견적입니다. 다시 조회해 주세요.')
    sid, q = QUOTES[inp.quote_id]
    eng = SESSIONS[sid]
    try:
        rec = eng.accept(q, inp.product)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {'booking': rec, 'summary': eng.summary(), 'session_id': sid}


@app.get('/summary')
def summary(session_id: Optional[str] = None):
    sid, eng = get_engine(session_id)
    s = eng.summary()
    s['container_cbm'] = bank.cbm
    s['session_id'] = sid
    s['policy_key'] = eng.policy.key
    return s


@app.get('/grid')
def grid(segment: int = 1, session_id: Optional[str] = None):
    """기업 모드용 요율표(히트맵). d*(판매시점 × 잔여용적)"""
    sid, eng = get_engine(session_id)
    g = eng.price_grid(segment)
    return {
        'segment': segment,
        't_bins': bank.T_BINS, 'cap_bins': bank.CAP_BINS,
        'rate_base': eng.r_S,
        'cells': [[{'discount': c['d'],
                    'offer': c['offer'],
                    'rate': round(eng.r_S * (1 - c['d']), 0),
                    'trained': c['trained'],
                    'visits': c['visits']} for c in row] for row in g],
    }


if __name__ == '__main__':
    import uvicorn
    # Render 등 클라우드는 PORT 환경변수로 포트를 지정합니다.
    # 로컬에서는 그냥 8000 번을 씁니다.
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
