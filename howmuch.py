"""
=============================================================================
 lcl_ask.py  --  물어보고 답하는 방식의 LCL 유연상품 추천운임 도구
=============================================================================
 학습은 하지 않는다.  train_policy_bank.py 가 만든
 results/policy_bank.json (80 cells = beta 4 x delta_F 2 x (9 season + ANY))
 을 읽어 학습된 d* 를 조회할 뿐이다.

 실행
   python lcl_ask.py
 그러면 화물 크기 / 가격민감도 / 수요국면 등을 하나씩 말로 물어본다.
 엔터만 치면 [ ] 안의 기본값을 쓴다.
=============================================================================
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

import lcl_core as L
from lcl_pricer import PolicyBank, UNIT_TO_DM

HERE = os.path.dirname(os.path.abspath(__file__))
BANK_PATH = os.path.join(HERE, 'results', 'policy_bank.json')

SELL_DAYS = 40.0            # 실제 판매기간(일)
WM_RULE = True              # W/M 과금: max(CBM, ton)
KR = {'Peak': '성수기', 'Normal': '일반기', 'OffPeak': '비성수기'}
KR2LV = {'성수기': 'Peak', '일반기': 'Normal', '일반': 'Normal',
         '비성수기': 'OffPeak', '1': 'Peak', '2': 'Normal', '3': 'OffPeak',
         'peak': 'Peak', 'normal': 'Normal', 'offpeak': 'OffPeak'}


# =============================================================================
# 물어보기
# =============================================================================
def ask(msg, default=None, cast=str, choices=None, hint=None):
    tail = '' if default is None else f'  [{default}]'
    if hint:
        print(f'  ({hint})')
    while True:
        raw = input(f'> {msg}{tail}\n  ').strip()
        if raw == '':
            if default is None and choices is None:
                return None
            return default
        try:
            v = cast(raw)
        except Exception:
            print('  ! 형식이 맞지 않습니다. 다시 입력해 주세요.\n')
            continue
        if choices and v not in choices:
            print(f'  ! {choices} 중에서 골라 주세요.\n')
            continue
        return v


def ask_yes(msg, default=True):
    d = 'Y/n' if default else 'y/N'
    r = input(f'> {msg}  [{d}]\n  ').strip().lower()
    if r == '':
        return default
    return r.startswith('y') or r.startswith('ㅇ')


def ask_dims():
    while True:
        raw = input('> 화물 한 개의 가로·세로·높이를 알려 주세요. '
                    '(공백으로 구분)\n  [12 8 6]\n  ').strip()
        raw = raw or '12 8 6'
        parts = raw.replace(',', ' ').replace('x', ' ').replace('X', ' ').split()
        try:
            v = [float(x) for x in parts]
        except ValueError:
            print('  ! 숫자 세 개를 입력해 주세요.\n')
            continue
        if len(v) != 3:
            print('  ! 숫자 세 개를 입력해 주세요.\n')
            continue
        return v


def ask_season():
    print('  (성수기 / 일반기 / 비성수기.  두 컨테이너가 다르면 '
          '"일반기/비성수기" 처럼 슬래시로)')
    while True:
        raw = input('> 이번 항차의 수요 국면은 어떻습니까?\n  [일반기]\n  ').strip()
        raw = raw or '일반기'
        parts = raw.replace('-', '/').split('/')
        if len(parts) == 1:
            parts = parts * 2
        if len(parts) != 2:
            print('  ! 형식: 일반기  또는  일반기/비성수기\n')
            continue
        out = []
        bad = False
        for x in parts:
            k = KR2LV.get(x.strip().lower(), KR2LV.get(x.strip()))
            if k is None:
                print(f'  ! "{x}" 를 모르겠습니다. 성수기/일반기/비성수기 중에서.\n')
                bad = True
                break
            out.append(k)
        if not bad:
            return tuple(out)


# =============================================================================
# 계산
# =============================================================================
def load_bank():
    if not os.path.exists(BANK_PATH):
        sys.exit(f'[!] policy_bank.json 을 찾지 못했습니다: {BANK_PATH}\n'
                 f'    먼저 python train_policy_bank.py 를 실행하세요.')
    return PolicyBank(BANK_PATH)


def synth_container(residual_cbm, seed=0):
    """잔여 용량만 알 때, 그 잔여를 재현하는 대표 적치를 시드 고정으로 합성."""
    c = L.EPContainer()
    target = max(0.0, L.CONT_CBM - float(residual_cbm))
    if target <= 1e-9:
        return c
    rng = np.random.default_rng(seed)
    for _ in range(4000):
        if c.used_cbm >= target - 1e-6:
            break
        v = min(rng.uniform(L.BOX_CBM_LOW, L.BOX_CBM_HIGH), target - c.used_cbm)
        if v <= 1e-3:
            break
        slot = c.find_slot(L.box_orientations(*L.make_box_dims(v, rng)))
        if slot is not None:
            c.commit(slot, v)
    return c


def price(bank, dims, unit, seg, days_left, c1, c2, season, beta, delta_F,
          r_S, weight_kg=None, seed=0, ep=True):
    k = UNIT_TO_DM[unit]
    d3 = tuple(float(x) * k for x in dims)
    cbm = float(np.prod(d3) / 1000.0)
    if WM_RULE and weight_kg:
        cbm = max(cbm, float(weight_kg) / 1000.0)

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

    pol = bank.select(beta, delta_F, season)
    tb = min(int(max(0.0, t) / T * bank.T_BINS), bank.T_BINS - 1)
    cb = lambda r: min(max(int(max(0.0, r) / cmax * bank.CAP_BINS), 0),
                       bank.CAP_BINS - 1)
    lk = pol.lookup(tb, cb(res[own]), cb(res[oth]))

    avail = ok[own] or ok[oth]
    offer = avail and lk['offer']
    d = lk['d'] if offer else 0.0
    assign = None
    if offer:
        order = [own, oth] if res[own] >= res[oth] else [oth, own]
        assign = next(i for i in order if ok[i])

    return {'cbm': cbm, 't': t, 'tb': tb, 'ob': cb(res[own]), 'xb': cb(res[oth]),
            'res': res, 'seg': seg, 'dims': list(dims), 'unit': unit,
            'days_left': days_left, 'season': season, 'beta': beta,
            'delta_F': delta_F, 'r_S': r_S,
            'S_ok': ok[own], 'S_rate': r_S, 'S_price': r_S * cbm,
            'S_reason': None if ok[own] else
                        ('부피 소진' if res[own] < cbm else '형상상 적재 불가'),
            'F_offer': offer, 'F_d': d, 'F_rate': r_S * (1 - d),
            'F_price': r_S * (1 - d) * cbm, 'F_save': r_S * d * cbm,
            'F_assign': None if assign is None else f'C{assign+1}',
            'F_reason': None if offer else
                        ('두 컨테이너 모두 적재 불가' if not avail
                         else '정책이 이 상태에서는 유연상품을 열지 않음'),
            'key': pol.key, 'action': lk['action_id'], 'trained': lk['trained'],
            'visits': lk['visits'], 'd_B': pol.d_static}


# =============================================================================
# 출력
# =============================================================================
def show(q):
    sea = '/'.join(KR[x] for x in q['season'])
    print()
    print('=' * 64)
    print(f"  화물   {q['dims']} {q['unit']}  = {q['cbm']:.3f} CBM"
          f"   희망 컨테이너 C{q['seg']}")
    print(f"  시점   마감 {q['days_left']:.0f}일 전  (t={q['t']:.1f}, "
          f"t_bin={q['tb']})")
    print(f"  잔여   C1 {q['res'][0]:.2f} / C2 {q['res'][1]:.2f} CBM"
          f"  (own_bin={q['ob']}, other_bin={q['xb']})")
    print(f"  국면   {sea}   민감도 beta={q['beta']}   "
          f"delta_F={q['delta_F']:+.2f}")
    print('-' * 64)
    if q['S_ok']:
        print(f"  [S] 표준  C{q['seg']}   {q['S_rate']:>12,.0f} /CBM"
              f"   {q['S_price']:>14,.0f}")
    else:
        print(f"  [S] 표준  불가 ({q['S_reason']})")
    if q['F_offer']:
        print(f"  [F] 유연  {q['F_assign']}   {q['F_rate']:>12,.0f} /CBM"
              f"   {q['F_price']:>14,.0f}   할인 d*={q['F_d']:.2f}")
        if q['S_ok']:
            print(f"      절감  {q['F_save']:>,.0f} ({q['F_d']*100:.0f}%)")
    else:
        print(f"  [F] 유연  미제공 ({q['F_reason']})")
    print('-' * 64)
    print(f"  policy {q['key']}   action={q['action']}  "
          f"{'trained' if q['trained'] else 'imputed'}(visits={q['visits']})"
          f"   d_B={q['d_B']}")
    print('=' * 64)


def cell(q):
    return '   미제공' if not q['F_offer'] else \
           f"{q['F_d']:.2f}/{q['F_price']:>9,.0f}"


def sensitivity(bank, base):
    kw = dict(base)
    print('\n[민감도 1] 가격민감도 beta x 수요국면        (셀 = d* / 유연운임)')
    print('  beta  ' + ''.join(f'{KR[s]:>18}' for s in L.LEVEL_NAMES))
    for b in L.BETA_LEVELS:
        row = [cell(price(bank, **{**kw, 'beta': b, 'season': (s, s)}))
               for s in L.LEVEL_NAMES]
        mk = '*' if abs(b - kw['beta']) < 1e-9 else ' '
        print(f'  {b:.1f}{mk} ' + ''.join(f'{c:>18}' for c in row))

    print('\n[민감도 2] 유연상품 선호도 delta_F')
    for dF in L.DELTA_F_LEVELS:
        print(f'  delta_F={dF:+.2f}    {cell(price(bank, **{**kw, "delta_F": float(dF)}))}')

    print('\n[민감도 3] 마감까지 남은 일수')
    for dl in (40, 30, 20, 10, 5, 0):
        q = price(bank, **{**kw, 'days_left': dl})
        print(f'  D-{dl:<3d}  t_bin={q["tb"]}    {cell(q)}')

    print('\n[민감도 4] 희망 컨테이너 잔여 용량')
    for fr in (1.0, 0.8, 0.6, 0.4, 0.2, 0.05):
        r = round(bank.cbm * fr, 2)
        k2 = {**kw, ('c1' if kw['seg'] == 1 else 'c2'): r}
        q = price(bank, **k2)
        print(f'  {r:>5.2f} CBM ({fr*100:>3.0f}%)  own_bin={q["ob"]}    {cell(q)}')

    print('\n[민감도 5] 상대 컨테이너 잔여 용량')
    for fr in (1.0, 0.8, 0.6, 0.4, 0.2, 0.05):
        r = round(bank.cbm * fr, 2)
        k2 = {**kw, ('c2' if kw['seg'] == 1 else 'c1'): r}
        q = price(bank, **k2)
        print(f'  {r:>5.2f} CBM ({fr*100:>3.0f}%)  other_bin={q["xb"]}    {cell(q)}')


def rate_card(bank, base):
    pol = bank.select(base['beta'], base['delta_F'], base['season'])
    q = price(bank, **base)
    print(f'\n[요율표] 학습된 d*   행=t_bin(0=판매개시 ~ {bank.T_BINS-1}=마감), '
          f'열=희망 컨테이너 잔여bin,  상대bin={q["xb"]} 고정')
    print('        ' + ''.join(f'{j:>7}' for j in range(bank.CAP_BINS)))
    for tb in range(bank.T_BINS):
        row = []
        for ob in range(bank.CAP_BINS):
            lk = pol.lookup(tb, ob, q['xb'])
            row.append('  no  ' if not lk['offer'] else f"{lk['d']:.2f}")
        print(f'  t={tb}  ' + ''.join(f'{c:>7}' for c in row))


# =============================================================================
def main():
    bank = load_bank()
    print('=' * 64)
    print('  LCL 유연상품 추천운임')
    print(f'  학습 정책 {len(bank.keys())}개 셀 로드  |  컨테이너 '
          f'2 x {bank.cbm:.2f} CBM  |  판매기간 {SELL_DAYS:.0f}일')
    print('  엔터만 치면 [ ] 안의 기본값을 사용합니다.')
    print('=' * 64)

    dims = ask_dims()
    unit = ask('길이 단위는 무엇입니까? (dm/cm/mm/m/in)', 'dm', str,
               list(UNIT_TO_DM))
    weight = ask('화물 중량은 몇 kg 입니까? (모르면 그냥 엔터 — 부피로만 과금)',
                 None, float)
    seg = ask('화주가 원하는 컨테이너는 1번입니까 2번입니까?', 1, int, [1, 2])
    days = ask('마감까지 며칠 남았습니까?', 20.0, float)
    c1 = ask('지금 C1 에 남은 용량은 몇 CBM 입니까?', round(bank.cbm, 2), float)
    c2 = ask('지금 C2 에 남은 용량은 몇 CBM 입니까?', round(bank.cbm, 2), float)
    season = ask_season()
    beta = ask('화주의 가격민감도 beta 는 얼마로 보십니까? (0.2 / 0.4 / 0.6 / 0.8)',
               L.BETA_BASE, float)
    dF = ask('유연상품 선호도 delta_F 는 얼마입니까? (0 = 표준과 동일, '
             '음수면 기피)', round(L.DELTA_F_BASE, 3), float)
    rs = ask('표준운임은 CBM 당 몇 원입니까?', 95000.0, float)

    base = dict(dims=dims, unit=unit, seg=seg, days_left=days, c1=c1, c2=c2,
                season=season, beta=beta, delta_F=dF, r_S=rs, weight_kg=weight)
    q = price(bank, **base)
    show(q)

    if ask_yes('민감도표도 보시겠습니까?', True):
        sensitivity(bank, base)
    if ask_yes('t x 잔여 요율표도 보시겠습니까?', False):
        rate_card(bank, base)

    if ask_yes('결과를 JSON 으로 저장할까요?', False):
        p = os.path.join(HERE, 'results', 'last_quote.json')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump({k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in q.items()}, open(p, 'w'), ensure_ascii=False,
                  indent=2)
        print(f'  저장 -> {p}')


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\n종료합니다.')