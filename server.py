"""
server.py  --  LCL 유연상품 추천운임 API
=============================================================================
 howmuch_final.py 의 price() 로직을 그대로 웹에서 부를 수 있게 감싼 것입니다.
 lcl_core.py / lcl_pricer.py / train_policy_bank.py 는 건드리지 않습니다.

 실행
   pip install -r requirements.txt
   python server.py            # http://localhost:8000/docs

 배포(Render)는 PORT 환경변수를 사용합니다.
 시황 브리핑을 쓰려면 환경변수 GEMINI_API_KEY 를 설정하세요.
=============================================================================
"""
from __future__ import annotations

import os
import re
import time
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import lcl_core as L
from lcl_pricer import UNIT_TO_DM, PolicyBank

# ---------------------------------------------------------------- 설정
HERE = os.path.dirname(os.path.abspath(__file__))
BANK_PATH = os.environ.get('POLICY_BANK',
                           os.path.join(HERE, 'results', 'policy_bank.json'))

SELL_DAYS = 40.0             # 실제 판매기간(일)
WM_RULE = True               # W/M 과금: max(부피CBM, 중량ton)

KR = {'Peak': '성수기', 'Normal': '일반기', 'OffPeak': '비성수기'}

bank = PolicyBank(BANK_PATH)

BETA_LEVELS = list(getattr(L, 'BETA_LEVELS', bank.betas))
DELTA_F_LEVELS = [float(x) for x in getattr(L, 'DELTA_F_LEVELS', bank.dFs)]
LEVEL_NAMES = list(getattr(L, 'LEVEL_NAMES', ['Peak', 'Normal', 'OffPeak']))
BETA_BASE = float(getattr(L, 'BETA_BASE', 0.6))
DELTA_F_BASE = float(getattr(L, 'DELTA_F_BASE', -0.3))
CONT_CBM = float(getattr(L, 'CONT_CBM', bank.cbm))

app = FastAPI(title='LCL 유연상품 추천운임 API', version='2.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_methods=['*'], allow_headers=['*'])


# ---------------------------------------------------------------- 적치 합성
_SYNTH_CACHE = {}


def synth_container(residual_cbm, seed=0):
    """잔여 용량만 알 때 그 잔여를 재현하는 대표 적치를 합성.
    howmuch_final.py 와 동일한 방식이며, 무거우므로 캐시합니다."""
    key = (round(float(residual_cbm), 2), int(seed))
    if key in _SYNTH_CACHE:
        return _SYNTH_CACHE[key]

    c = L.EPContainer()
    target = max(0.0, CONT_CBM - float(residual_cbm))
    if target > 1e-9:
        rng = np.random.default_rng(seed)
        lo = float(getattr(L, 'BOX_CBM_LOW', 0.3))
        hi = float(getattr(L, 'BOX_CBM_HIGH', 2.0))
        for _ in range(4000):
            if c.used_cbm >= target - 1e-6:
                break
            v = min(rng.uniform(lo, hi), target - c.used_cbm)
            if v <= 1e-3:
                break
            slot = c.find_slot(L.box_orientations(*L.make_box_dims(v, rng)))
            if slot is not None:
                c.commit(slot, v)
    _SYNTH_CACHE[key] = c
    return c


# ---------------------------------------------------------------- 가격 산출
def season_tuple(season):
    """'Normal-OffPeak' | ('Normal','OffPeak') | '일반기/비성수기' -> 튜플"""
    if season is None:
        return None
    if isinstance(season, (list, tuple)):
        parts = list(season)
    else:
        s = str(season).strip()
        if s.upper() == 'ANY':
            return None
        parts = re.split(r'[-/]', s)
    if len(parts) == 1:
        parts = parts * 2
    inv = {v: k for k, v in KR.items()}
    return tuple(inv.get(str(p).strip(), str(p).strip()) for p in parts[:2])


def price(dims, unit, seg, days_left, c1, c2, season, beta, delta_F,
          r_S=None, weight_kg=None, seed=0, ep=True):
    """howmuch_final.price() 와 동일한 계산."""
    k = UNIT_TO_DM[unit]
    d3 = tuple(float(x) * k for x in dims)
    vol_cbm = float(np.prod(d3) / 1000.0)
    cbm, basis = vol_cbm, 'volume'
    if WM_RULE and weight_kg and float(weight_kg) / 1000.0 > cbm:
        cbm, basis = float(weight_kg) / 1000.0, 'weight'

    T, cmax = bank.T, bank.cbm
    t = (1.0 - min(max(float(days_left), 0.0), SELL_DAYS) / SELL_DAYS) * T

    own, oth = seg - 1, 2 - seg
    res = [float(c1), float(c2)]

    if ep:
        cs = [synth_container(res[0], seed), synth_container(res[1], seed + 1)]
        ors = L.box_orientations(*d3)
        ok = [res[i] >= cbm and cs[i].find_slot(ors) is not None for i in (0, 1)]
    else:
        ok = [res[i] >= cbm for i in (0, 1)]

    pol = bank.select(beta, delta_F, season_tuple(season))
    tb = min(int(max(0.0, t) / T * bank.T_BINS), bank.T_BINS - 1)

    def cb(r):
        return min(max(int(max(0.0, r) / cmax * bank.CAP_BINS), 0),
                   bank.CAP_BINS - 1)

    lk = pol.lookup(tb, cb(res[own]), cb(res[oth]))

    avail = ok[own] or ok[oth]
    offer = bool(avail and lk['offer'])
    d = float(lk['d']) if offer else 0.0
    assign = None
    if offer:
        order = [own, oth] if res[own] >= res[oth] else [oth, own]
        assign = next(i for i in order if ok[i])

    rate = None if r_S in (None, '') else float(r_S)
    money = rate is not None
    st = season_tuple(season) or ('ANY', 'ANY')

    return {
        'cargo': {'dims': list(dims), 'unit': unit,
                  'dims_dm': [round(x, 3) for x in d3],
                  'volume_cbm': round(vol_cbm, 4),
                  'billed_cbm': round(cbm, 4),
                  'weight_ton': round((weight_kg or 0) / 1000.0, 4),
                  'basis': basis},
        'timing': {'days_left': float(days_left), 'sell_days': SELL_DAYS,
                   't': round(t, 2), 't_bin': tb, 't_bins': bank.T_BINS},
        'capacity': {'residual_cbm': [round(res[0], 2), round(res[1], 2)],
                     'container_cbm': round(cmax, 2),
                     'own_bin': cb(res[own]), 'other_bin': cb(res[oth]),
                     'cap_bins': bank.CAP_BINS,
                     'load_factor': [round(1 - res[0] / cmax, 3),
                                     round(1 - res[1] / cmax, 3)]},
        'segment': seg,
        'standard': {
            'available': bool(ok[own]),
            'container': 'C%d' % seg,
            'multiplier': 1.0,
            'rate_per_cbm': rate,
            'price': round(rate * cbm) if money else None,
            'reason': None if ok[own] else
                      ('부피 소진' if res[own] < cbm else '형상상 적재 불가'),
        },
        'flexible': {
            'offered': offer,
            'discount': round(d, 3),
            'multiplier': round(1.0 - d, 3),
            'assign_preview': None if assign is None else 'C%d' % (assign + 1),
            'rate_per_cbm': round(rate * (1 - d)) if money else None,
            'price': round(rate * (1 - d) * cbm) if money else None,
            'saving': round(rate * d * cbm) if money else None,
            'saving_cbm': round(d * cbm, 4),
            'reason': None if offer else
                      ('두 컨테이너 모두 적재 불가' if not avail
                       else '정책이 이 상태에서는 유연상품을 열지 않음'),
        },
        'policy': {
            'key': pol.key, 'action_id': lk['action_id'],
            'trained': bool(lk['trained']), 'visits': int(lk['visits']),
            'd_static_benchmark': pol.d_static,
            'beta': beta, 'delta_F': delta_F,
            'season': list(st),
            'season_kr': [KR.get(x, '구분없음') for x in st],
        },
        'money_mode': money,
    }


# ---------------------------------------------------------------- 스키마
class QuoteIn(BaseModel):
    length: float = Field(12.0, gt=0)
    width: float = Field(8.0, gt=0)
    height: float = Field(6.0, gt=0)
    unit: str = Field('cm', description='dm/cm/mm/m/in')
    weight_kg: Optional[float] = Field(None, ge=0)
    segment: int = Field(1, ge=1, le=2)
    days_left: float = Field(20.0, ge=0)
    c1: Optional[float] = Field(None, description='C1 잔여 CBM')
    c2: Optional[float] = Field(None, description='C2 잔여 CBM')
    season: str = Field('Normal-Normal')
    beta: float = Field(BETA_BASE)
    delta_F: float = Field(DELTA_F_BASE)
    r_S: Optional[float] = Field(None, description='표준 구간운임(원/CBM)')
    ep: bool = Field(True)

    def kw(self):
        return dict(dims=[self.length, self.width, self.height], unit=self.unit,
                    seg=self.segment, days_left=self.days_left,
                    c1=bank.cbm if self.c1 is None else self.c1,
                    c2=bank.cbm if self.c2 is None else self.c2,
                    season=self.season, beta=self.beta, delta_F=self.delta_F,
                    r_S=self.r_S, weight_kg=self.weight_kg, ep=self.ep)


# ---------------------------------------------------------------- 엔드포인트
@app.get('/', response_class=HTMLResponse)
def index():
    return """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>LCL 유연상품 추천운임 API</title><style>
body{font-family:-apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo',sans-serif;
max-width:660px;margin:60px auto;padding:0 20px;line-height:1.75;color:#16202e}
h1{font-size:22px;margin-bottom:2px}p.sub{color:#67728a;margin-top:0;font-size:14px}
code{background:#f2f5f9;padding:2px 6px;border-radius:4px;font-size:13px}
li{margin:5px 0}a{color:#2b5cd9}</style></head><body>
<h1>LCL 유연상품 추천운임 API</h1>
<p class="sub">강화학습 기반 할인정책 조회 &middot; 백엔드 서버</p>
<p>여기는 API 서버입니다. 사용자 화면은 아래에서 확인하세요.</p>
<p>&rarr; <a href="https://lcl-ocean-freight-dynamic-pricing.ai.studio">서비스 바로가기</a></p>
<h3>엔드포인트</h3><ul>
<li><code>GET /meta</code> 정책 축 &middot; 시즌 &middot; 기본값</li>
<li><code>POST /quote</code> 표준 / 유연 운임 산출</li>
<li><code>POST /sensitivity</code> 민감도 5종</li>
<li><code>POST /ratecard</code> 시점 &times; 잔여용적 요율표</li>
<li><code>GET /market</code> 시황 브리핑</li></ul>
<p>문서: <a href="/docs">/docs</a></p></body></html>"""


@app.get('/meta')
def meta():
    return {
        'axes': {'beta': BETA_LEVELS, 'delta_F': DELTA_F_LEVELS},
        'season_levels': [{'value': v, 'label': KR[v]} for v in LEVEL_NAMES],
        'seasons': sorted({k.rsplit('|', 1)[-1] for k in bank.keys()}),
        'defaults': {'beta': BETA_BASE, 'delta_F': DELTA_F_BASE,
                     'season': 'Normal-Normal', 'unit': 'cm',
                     'days_left': 20.0, 'segment': 1},
        'sell_days': SELL_DAYS,
        'container_cbm': round(bank.cbm, 2),
        'bins': {'t_bins': bank.T_BINS, 'cap_bins': bank.CAP_BINS},
        'discounts': sorted({a.get('d', 0.0) for a in bank.actions}),
        'units': list(UNIT_TO_DM),
        'policy_cells': len(bank.keys()),
        'wm_rule': WM_RULE,
    }


@app.post('/quote')
def quote(inp: QuoteIn):
    try:
        return price(**inp.kw())
    except KeyError as e:
        raise HTTPException(400, '해당 조건의 학습 정책이 없습니다: %s' % e)
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(400, '견적 실패: %s' % e)


@app.post('/sensitivity')
def sensitivity(inp: QuoteIn):
    """할인율 d* 가 무엇에 따라 움직이는지 보여주는 5종 민감도."""
    kw = inp.kw()

    def cell(q):
        return {'offered': q['flexible']['offered'],
                'discount': q['flexible']['discount'],
                'multiplier': q['flexible']['multiplier'],
                'price': q['flexible']['price'],
                'trained': q['policy']['trained']}

    try:
        t1 = [{'beta': b, 'current': abs(b - kw['beta']) < 1e-9,
               'cells': [cell(price(**{**kw, 'beta': b, 'season': (s, s)}))
                         for s in LEVEL_NAMES]}
              for b in BETA_LEVELS]

        t2 = [{'delta_F': float(x), 'current': abs(x - kw['delta_F']) < 1e-9,
               **cell(price(**{**kw, 'delta_F': float(x)}))}
              for x in DELTA_F_LEVELS]

        t3 = []
        for dl in (40, 30, 20, 10, 5, 0):
            q = price(**{**kw, 'days_left': dl})
            t3.append({'days_left': dl, 't_bin': q['timing']['t_bin'],
                       'current': abs(dl - kw['days_left']) < 1e-9, **cell(q)})

        key_own = 'c1' if kw['seg'] == 1 else 'c2'
        key_oth = 'c2' if kw['seg'] == 1 else 'c1'
        t4, t5 = [], []
        for fr in (1.0, 0.8, 0.6, 0.4, 0.2, 0.05):
            r = round(bank.cbm * fr, 2)
            q4 = price(**{**kw, key_own: r})
            t4.append({'residual_cbm': r, 'ratio': fr,
                       'bin': q4['capacity']['own_bin'], **cell(q4)})
            q5 = price(**{**kw, key_oth: r})
            t5.append({'residual_cbm': r, 'ratio': fr,
                       'bin': q5['capacity']['other_bin'], **cell(q5)})
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(400, '민감도 산출 실패: %s' % e)

    return {
        'season_levels': [{'value': v, 'label': KR[v]} for v in LEVEL_NAMES],
        'beta_x_season': t1, 'delta_F': t2, 'days_left': t3,
        'own_residual': t4, 'other_residual': t5,
        'base': {'beta': kw['beta'], 'delta_F': kw['delta_F'],
                 'days_left': kw['days_left'], 'segment': kw['seg'],
                 'r_S': kw['r_S']},
    }


@app.post('/ratecard')
def ratecard(inp: QuoteIn):
    """행 = 판매시점 구간, 열 = 희망 컨테이너 잔여용적 구간."""
    kw = inp.kw()
    q = price(**kw)
    pol = bank.select(kw['beta'], kw['delta_F'], season_tuple(kw['season']))
    xb = q['capacity']['other_bin']
    rows = []
    for tb in range(bank.T_BINS):
        row = []
        for ob in range(bank.CAP_BINS):
            lk = pol.lookup(tb, ob, xb)
            row.append({'offer': lk['offer'], 'discount': round(lk['d'], 3),
                        'multiplier': round(1 - lk['d'], 3),
                        'trained': bool(lk['trained']),
                        'visits': int(lk['visits']),
                        'current': (tb == q['timing']['t_bin']
                                    and ob == q['capacity']['own_bin'])})
        rows.append(row)
    return {'t_bins': bank.T_BINS, 'cap_bins': bank.CAP_BINS,
            'other_bin_fixed': xb, 'policy_key': pol.key,
            'd_static_benchmark': pol.d_static,
            'current': {'t_bin': q['timing']['t_bin'],
                        'own_bin': q['capacity']['own_bin']},
            'cells': rows}


# ---------------------------------------------------------------- 시황 브리핑
GEMINI_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')
_MARKET = {'at': 0.0, 'data': None}
_MARKET_TTL = 1800

_PROMPT = """당신은 해상운송 시장 애널리스트입니다.
Google 검색으로 최근 정보를 찾아 한국발 LCL(소량화물) 해상운송 시장을 브리핑하세요.

아래 JSON 형식으로만 답하세요. 마크다운 코드펜스나 설명 없이 JSON 만 출력합니다.

{
  "headline": "한 문장 시황 요약",
  "sentiment": "상승" 또는 "보합" 또는 "하락",
  "drivers": ["운임에 영향을 주는 요인 3가지, 각 25자 이내"],
  "implication": "동적 가격결정 관점의 시사점 두 문장",
  "news": [{"title":"기사를 직접 요약한 제목","source":"매체명",
            "date":"YYYY-MM-DD","note":"왜 중요한지 한 문장","url":"원문 링크"}]
}

규칙: news 는 3~5건, 최근 것 우선. 기사 원문 문장을 그대로 옮기지 말고
반드시 자신의 말로 바꿔 쓰세요. 확인되지 않은 수치는 쓰지 마세요. 한국어로 작성하세요."""


def _gemini(prompt):
    import json as _json
    import urllib.request
    url = ('https://generativelanguage.googleapis.com/v1beta/models/'
           + GEMINI_MODEL + ':generateContent?key=' + GEMINI_KEY)
    body = {'contents': [{'parts': [{'text': prompt}]}],
            'tools': [{'google_search': {}}]}
    req = urllib.request.Request(
        url, data=_json.dumps(body).encode('utf-8'),
        headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=60) as r:
        payload = _json.loads(r.read().decode('utf-8'))
    parts = payload['candidates'][0]['content']['parts']
    return ''.join(p.get('text', '') for p in parts)


def _parse_json_block(text):
    import json as _json
    t = text.strip()
    if t.startswith('```'):
        t = re.sub(r'^```[a-zA-Z]*\n?', '', t)
        t = re.sub(r'\n?```$', '', t.strip())
    i, j = t.find('{'), t.rfind('}')
    if i >= 0 and j > i:
        t = t[i:j + 1]
    return _json.loads(t)


@app.get('/market')
def market(refresh: bool = False):
    if not GEMINI_KEY:
        return {'available': False,
                'reason': 'GEMINI_API_KEY 가 서버에 설정되지 않았습니다.'}
    now = time.time()
    if not refresh and _MARKET['data'] and now - _MARKET['at'] < _MARKET_TTL:
        return dict(_MARKET['data'], cached=True)
    try:
        data = _parse_json_block(_gemini(_PROMPT))
    except Exception as e:                                   # noqa: BLE001
        if _MARKET['data']:
            return dict(_MARKET['data'], cached=True, stale=True)
        return {'available': False, 'reason': '브리핑 생성 실패: %s' % e}
    data.update(available=True, cached=False, model=GEMINI_MODEL,
                generated_at=time.strftime('%Y-%m-%d %H:%M',
                                           time.localtime(now)))
    _MARKET.update(at=now, data=data)
    return data


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
