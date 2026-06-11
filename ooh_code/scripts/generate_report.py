import numpy as np, re, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

def parse_log(logpath):
    with open(logpath, 'r', encoding='utf-8') as f:
        text = f.read()
    r = {}
    pats = {
        'avg_charge': r'Avg\. Charge:\s+([\d.]+)',
        'avg_charge_std': r'Avg\. Charge:\s+[\d.]+\s+std\.:\s+([\d.]+)',
        'avg_discount': r'Avg\. Discount:\s+([\d.]+)',
        'avg_discount_std': r'Avg\. Discount:\s+[\d.]+\s+std\.:\s+([\d.]+)',
        'charge_rev': r'Charge revenue:\s+([\d.]+)',
        'discount_costs': r'Discount costs:\s+([\d.]+)',
        'base_rev': r'Base revenue:\s+([\d.]+)',
        'net_profit': r'Net profit:\s+([\d.]+)',
        'total_costs': r'total costs:\s+([\d.]+)',
        'home_pct': r'percentage home delivery:\s+([\d.]+)',
        'travel_costs': r'travel costs:\s+([\d.]+)',
        'service_costs': r'service costs:\s+([\d.]+)',
        'failure_costs': r'failure costs:\s+([\d.]+)',
    }
    for key, pat in pats.items():
        m = re.search(pat, text)
        r[key] = float(m.group(1)) if m else 0.0
    m = re.search(r'Quit rate:\s+([\d.]+)%', text)
    r['quit_rate'] = float(m.group(1)) if m else 0.0
    m = re.search(r'Total customers:\s+(\d+)', text)
    r['total_customers'] = int(m.group(1)) if m else 0
    m = re.search(r'total timing:\s+([\d.]+)', text)
    r['total_time'] = float(m.group(1)) if m else 0.0
    return r

lt = {
    'Base': {
        'DSPO': 'Experiments/Parcelpoint_py/pricing/DSPO/YJ_BASE_RC_400_DSPO_seed{}_base_rc/{}/Logs/logfile.log',
        'DRPO': 'Experiments/Parcelpoint_py/pricing/DRPO/YJ_BASE_RC_400_DRPO_seed{}_base_rc/{}/Logs/logfile.log',
    },
    'Dispersed': {
        'DSPO': 'Experiments/Parcelpoint_py/pricing/DSPO/YJ_DISP_RC_400_DSPO_seed{}_disp_rc/{}/Logs/logfile.log',
        'DRPO': 'Experiments/Parcelpoint_py/pricing/DRPO/YJ_DISP_RC_400_DRPO_seed{}_disp_rc/{}/Logs/logfile.log',
    },
}

tc_lt = {
    'Base': {
        'DSPO': 'Experiments/Parcelpoint_py/pricing/DSPO/YJ_BASE_RC_400_DSPO_seed{}_base_rc/{}/Results/training_curve.npy',
        'DRPO': 'Experiments/Parcelpoint_py/pricing/DRPO/YJ_BASE_RC_400_DRPO_seed{}_base_rc/{}/Results/training_curve.npy',
    },
    'Dispersed': {
        'DSPO': 'Experiments/Parcelpoint_py/pricing/DSPO/YJ_DISP_RC_400_DSPO_seed{}_disp_rc/{}/Results/training_curve.npy',
        'DRPO': 'Experiments/Parcelpoint_py/pricing/DRPO/YJ_DISP_RC_400_DRPO_seed{}_disp_rc/{}/Results/training_curve.npy',
    },
}

seeds = [40, 67, 97]
ar = {}
for inst in ['Base', 'Dispersed']:
    ar[inst] = {}
    for algo in ['DSPO', 'DRPO']:
        ar[inst][algo] = {}
        for seed in seeds:
            path = lt[inst][algo].format(seed, seed)
            ar[inst][algo][seed] = parse_log(path)

lines = []
L = lines.append

L('=' * 100)
L('  YANJIAO DRT EXPERIMENT: COMPLETE RESULTS REPORT')
L('  DRPO (SPO+) vs DSPO (MSE) with RC-Matched Parameters')
L('  Generated: 2026-05-31')
L('=' * 100)
L('')
L('1. EXPERIMENT CONFIGURATION')
L('-' * 100)
L('  max_price=2.0, min_price=-10.0, revenue=50, fuel_cost=0.6')
L('  incentive_sens=-0.25, home_util=1.4, base_util=-1.0, outside_option_util=-1.0')
L('  walk_distance_weight=0.0, travel_time_weight=0.0, use_travel_time_prediction=False')
L('  n_passengers=400, n_vehicles=35, veh_capacity=12, k=10')
L('  episodes=200, eval_episodes=20, seeds=[40, 67, 97]')
L('  DSPO: spo_loss_weight=0.0 (pure MSE)  |  DRPO: spo_loss_weight=0.7 (MSE + SPO+)')
L('')

L('2. FULL RESULTS: BASE INSTANCE')
L('-' * 100)
hdr = '{:>5} {:>5} | {:>10} {:>10} {:>9} {:>8} {:>8} | {:>6} {:>6} | {:>7} {:>7} {:>7} {:>7} | {:>7} {:>8} {:>8} | {:>7}'.format(
    'Seed','Algo','NetProfit','TotalCost','TravCost','SvcCost','FailCost','Home%','Quit%','AvgChg','ChgStd','AvgDsc','DscStd','ChgRev','DscCost','BaseRev','Time')
L(hdr)
L('-' * len(hdr))

for seed in seeds:
    for algo in ['DSPO', 'DRPO']:
        r = ar['Base'][algo][seed]
        L('{:>5} {:>5} | {:>10.1f} {:>10.1f} {:>9.1f} {:>8.1f} {:>8.1f} | {:>5.1f}% {:>5.2f}% | {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f} | {:>7.1f} {:>8.1f} {:>8.1f} | {:>6.0f}s'.format(
            seed, algo, r['net_profit'], r['total_costs'], r['travel_costs'], r['service_costs'], r['failure_costs'],
            r['home_pct']*100, r['quit_rate'], r['avg_charge'], r['avg_charge_std'], r['avg_discount'], r['avg_discount_std'],
            r['charge_rev'], r['discount_costs'], r['base_rev'], r['total_time']))
    L('')

L('  MEAN:')
for algo in ['DSPO', 'DRPO']:
    ms = {}
    for k in ['net_profit','total_costs','travel_costs','service_costs','failure_costs','home_pct','quit_rate','avg_charge','avg_charge_std','avg_discount','avg_discount_std','charge_rev','discount_costs','base_rev','total_time']:
        ms[k] = np.mean([ar['Base'][algo][s][k] for s in seeds])
    L('{:>5} {:>5} | {:>10.1f} {:>10.1f} {:>9.1f} {:>8.1f} {:>8.1f} | {:>5.1f}% {:>5.2f}% | {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f} | {:>7.1f} {:>8.1f} {:>8.1f} | {:>6.0f}s'.format(
        '', algo, ms['net_profit'], ms['total_costs'], ms['travel_costs'], ms['service_costs'], ms['failure_costs'],
        ms['home_pct']*100, ms['quit_rate'], ms['avg_charge'], ms['avg_charge_std'], ms['avg_discount'], ms['avg_discount_std'],
        ms['charge_rev'], ms['discount_costs'], ms['base_rev'], ms['total_time']))

L('')
L('  PAIRED DELTA (DRPO - DSPO):')
wins = 0
for seed in seeds:
    d = ar['Base']['DSPO'][seed]
    r = ar['Base']['DRPO'][seed]
    dp = r['net_profit'] - d['net_profit']
    dh = (r['home_pct'] - d['home_pct']) * 100
    dc = r['total_costs'] - d['total_costs']
    if dp > 0: wins += 1
    L('    seed={}: profit={:+.1f}, costs={:+.1f}, home%={:+.1f}pp'.format(seed, dp, dc, dh))
dm = np.mean([ar['Base']['DSPO'][s]['net_profit'] for s in seeds])
rm = np.mean([ar['Base']['DRPO'][s]['net_profit'] for s in seeds])
L('    MEAN delta: {:+.1f}, DRPO wins: {}/3'.format(rm-dm, wins))

L('')
L('3. FULL RESULTS: DISPERSED INSTANCE')
L('-' * 100)
L(hdr)
L('-' * len(hdr))

for seed in seeds:
    for algo in ['DSPO', 'DRPO']:
        r = ar['Dispersed'][algo][seed]
        L('{:>5} {:>5} | {:>10.1f} {:>10.1f} {:>9.1f} {:>8.1f} {:>8.1f} | {:>5.1f}% {:>5.2f}% | {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f} | {:>7.1f} {:>8.1f} {:>8.1f} | {:>6.0f}s'.format(
            seed, algo, r['net_profit'], r['total_costs'], r['travel_costs'], r['service_costs'], r['failure_costs'],
            r['home_pct']*100, r['quit_rate'], r['avg_charge'], r['avg_charge_std'], r['avg_discount'], r['avg_discount_std'],
            r['charge_rev'], r['discount_costs'], r['base_rev'], r['total_time']))
    L('')

L('  MEAN:')
for algo in ['DSPO', 'DRPO']:
    ms = {}
    for k in ['net_profit','total_costs','travel_costs','service_costs','failure_costs','home_pct','quit_rate','avg_charge','avg_charge_std','avg_discount','avg_discount_std','charge_rev','discount_costs','base_rev','total_time']:
        ms[k] = np.mean([ar['Dispersed'][algo][s][k] for s in seeds])
    L('{:>5} {:>5} | {:>10.1f} {:>10.1f} {:>9.1f} {:>8.1f} {:>8.1f} | {:>5.1f}% {:>5.2f}% | {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f} | {:>7.1f} {:>8.1f} {:>8.1f} | {:>6.0f}s'.format(
        '', algo, ms['net_profit'], ms['total_costs'], ms['travel_costs'], ms['service_costs'], ms['failure_costs'],
        ms['home_pct']*100, ms['quit_rate'], ms['avg_charge'], ms['avg_charge_std'], ms['avg_discount'], ms['avg_discount_std'],
        ms['charge_rev'], ms['discount_costs'], ms['base_rev'], ms['total_time']))

L('')
L('  PAIRED DELTA (DRPO - DSPO):')
wins = 0
for seed in seeds:
    d = ar['Dispersed']['DSPO'][seed]
    r = ar['Dispersed']['DRPO'][seed]
    dp = r['net_profit'] - d['net_profit']
    dh = (r['home_pct'] - d['home_pct']) * 100
    dc = r['total_costs'] - d['total_costs']
    if dp > 0: wins += 1
    L('    seed={}: profit={:+.1f}, costs={:+.1f}, home%={:+.1f}pp'.format(seed, dp, dc, dh))
dm = np.mean([ar['Dispersed']['DSPO'][s]['net_profit'] for s in seeds])
rm = np.mean([ar['Dispersed']['DRPO'][s]['net_profit'] for s in seeds])
L('    MEAN delta: {:+.1f}, DRPO wins: {}/3'.format(rm-dm, wins))

L('')
L('4. CROSS-INSTANCE SUMMARY')
L('-' * 100)
L('{:>12} {:>12} {:>12} {:>8} {:>6} {:>11} {:>11} {:>11} {:>11}'.format(
    'Instance','DSPO Profit','DRPO Profit','Delta','Wins','DSPO Home%','DRPO Home%','DSPO Quit%','DRPO Quit%'))
L('-' * 100)
for inst in ['Base', 'Dispersed']:
    dp = [ar[inst]['DSPO'][s]['net_profit'] for s in seeds]
    rp = [ar[inst]['DRPO'][s]['net_profit'] for s in seeds]
    dh = [ar[inst]['DSPO'][s]['home_pct']*100 for s in seeds]
    rh = [ar[inst]['DRPO'][s]['home_pct']*100 for s in seeds]
    dq = [ar[inst]['DSPO'][s]['quit_rate'] for s in seeds]
    rq = [ar[inst]['DRPO'][s]['quit_rate'] for s in seeds]
    w = sum(1 for a,b in zip(dp,rp) if b>a)
    L('{:>12} {:>12.1f} {:>12.1f} {:>+8.1f} {}/3    {:>10.1f}% {:>10.1f}% {:>10.2f}% {:>10.2f}%'.format(
        inst, np.mean(dp), np.mean(rp), np.mean(rp)-np.mean(dp), w, np.mean(dh), np.mean(rh), np.mean(dq), np.mean(rq)))

all_dp = [ar[i]['DSPO'][s]['net_profit'] for i in ['Base','Dispersed'] for s in seeds]
all_rp = [ar[i]['DRPO'][s]['net_profit'] for i in ['Base','Dispersed'] for s in seeds]
all_w = sum(1 for a,b in zip(all_dp, all_rp) if b>a)
L('{:>12} {:>12.1f} {:>12.1f} {:>+8.1f} {}/6'.format(
    'Combined', np.mean(all_dp), np.mean(all_rp), np.mean(all_rp)-np.mean(all_dp), all_w))

L('')
L('5. RC BENCHMARK REFERENCE (30 seeds)')
L('-' * 100)
L('  DSPO: profit=2193.8, costs=2218.5, home%=53.66%, quit%=3.69%')
L('  DRPO: profit=2346.1, costs=2093.1, home%=45.32%, quit%=3.11%')
L('  Delta: +152.3 (+6.9%), DRPO wins: 29/30')
L('  95% CI: +/-38.66 (significant)')

L('')
L('6. TRAINING CURVE ANALYSIS')
L('-' * 100)
L('{:>12} {:>5} {:>5} | {:>7} {:>7} {:>7} {:>7} {:>7} | {:>10} {:>10}'.format(
    'Instance','Seed','Algo','Ep0','Ep10','Ep50','Ep100','Ep199','Last50Mean','Last50Std'))
L('-' * 100)

for inst in ['Base', 'Dispersed']:
    for seed in seeds:
        for algo in ['DSPO', 'DRPO']:
            tc = np.load(tc_lt[inst][algo].format(seed, seed))
            l50m = np.mean(tc[-50:])
            l50s = np.std(tc[-50:])
            L('{:>12} {:>5} {:>5} | {:>7.0f} {:>7.0f} {:>7.0f} {:>7.0f} {:>7.0f} | {:>10.1f} {:>10.1f}'.format(
                inst, seed, algo, tc[0], tc[10], tc[50], tc[100], tc[199], l50m, l50s))
    L('')

L('7. COST DECOMPOSITION (avg over 3 seeds)')
L('-' * 100)
L('{:>20} {:>12} {:>12} {:>8} | {:>12} {:>12} {:>8}'.format(
    'Component','Base DSPO','Base DRPO','Delta','Disp DSPO','Disp DRPO','Delta'))
L('-' * 100)
for comp in ['travel_costs','service_costs','failure_costs','discount_costs','charge_rev']:
    bv = [np.mean([ar['Base'][a][s][comp] for s in seeds]) for a in ['DSPO','DRPO']]
    dv = [np.mean([ar['Dispersed'][a][s][comp] for s in seeds]) for a in ['DSPO','DRPO']]
    L('{:>20} {:>12.1f} {:>12.1f} {:>+8.1f} | {:>12.1f} {:>12.1f} {:>+8.1f}'.format(
        comp, bv[0], bv[1], bv[1]-bv[0], dv[0], dv[1], dv[1]-dv[0]))

L('-' * 100)
for label, keys in [('Total Ops',['travel_costs','service_costs','failure_costs']),
                     ('Net Pricing',['charge_rev','discount_costs'])]:
    bv = [sum(np.mean([ar['Base'][a][s][k] for s in seeds]) for k in keys) for a in ['DSPO','DRPO']]
    dv = [sum(np.mean([ar['Dispersed'][a][s][k] for s in seeds]) for k in keys) for a in ['DSPO','DRPO']]
    L('{:>20} {:>12.1f} {:>12.1f} {:>+8.1f} | {:>12.1f} {:>12.1f} {:>+8.1f}'.format(
        label, bv[0], bv[1], bv[1]-bv[0], dv[0], dv[1], dv[1]-dv[0]))

L('')
L('8. PRICING STRATEGY ANALYSIS')
L('-' * 100)
L('  Base:   DRPO and DSPO produce IDENTICAL pricing (AvgCharge diff<0.01, Home% identical 12.3%)')
L('  Dispersed: DRPO gives deeper discounts (+0.12~0.18), fewer home pickups (15.6% vs 17.6%)')
L('  DRPO directionally correct but magnitude limited by low heterogeneity')

L('')
L('9. WHY DRPO ADVANTAGE IS LIMITED ON YANJIAO')
L('-' * 100)
L('  A) Low capacity pressure: 400p/35v/cap12=420 slots, quit<1%')
L('  B) Low home pickup: 12-18% vs RC 45-54% (pricing affects few customers)')
L('  C) Low MP cost heterogeneity: real bus stops along fixed route')
L('  D) MSE already accurate: training curves differ <0.3%')

L('')
L('=' * 100)
L('END OF REPORT')
L('=' * 100)

outpath = 'Experiments/analysis/yanjiao_complete_report.txt'
with open(outpath, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print('Report saved to: ' + outpath)
print()
for l in lines:
    print(l)
