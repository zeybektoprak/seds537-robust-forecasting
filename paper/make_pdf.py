"""Generate the project report PDF using reportlab.

Usage:  python3 make_pdf.py
Output: Zeybek_SEDS537_RobustForecasting.pdf
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from pathlib import Path

ROOT    = Path(__file__).parent.parent
FIG_DIR = ROOT / 'results' / 'figures'
OUT     = Path(__file__).parent / 'Zeybek_SEDS537_RobustForecasting.pdf'

DATASET_URL = "https://raw.githubusercontent.com/gilbutITbook/006975/master/datasets/jena_climate/jena_climate_2009_2016.csv"
GITHUB_URL  = "https://github.com/zeybektoprak/seds537-robust-forecasting"

W, H = A4
styles = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, parent=styles['Normal'], **kw)

title_style   = S('Title',  fontSize=16, leading=20, alignment=TA_CENTER, spaceAfter=6, fontName='Helvetica-Bold')
author_style  = S('Author', fontSize=11, leading=14, alignment=TA_CENTER, spaceAfter=4)
inst_style    = S('Inst',   fontSize=9,  leading=12, alignment=TA_CENTER, spaceAfter=2, textColor=colors.grey)
h1_style      = S('H1',     fontSize=13, leading=16, fontName='Helvetica-Bold', spaceBefore=14, spaceAfter=4)
h2_style      = S('H2',     fontSize=11, leading=14, fontName='Helvetica-Bold', spaceBefore=8,  spaceAfter=3)
h3_style      = S('H3',     fontSize=10, leading=13, fontName='Helvetica-BoldOblique', spaceBefore=6, spaceAfter=2)
body_style    = S('Body',   fontSize=9.5,leading=14, alignment=TA_JUSTIFY, spaceAfter=4)
bullet_style  = S('Bullet', fontSize=9.5,leading=13, alignment=TA_JUSTIFY, leftIndent=14, bulletIndent=4, spaceAfter=2)
cap_style     = S('Cap',    fontSize=8.5,leading=11, alignment=TA_CENTER, textColor=colors.HexColor('#444444'), spaceAfter=8)
abstract_style= S('Abs',    fontSize=9,  leading=13, alignment=TA_JUSTIFY, leftIndent=20, rightIndent=20, spaceAfter=6)
kw_style      = S('KW',     fontSize=8.5,leading=12, alignment=TA_JUSTIFY, leftIndent=20, rightIndent=20, spaceAfter=10, textColor=colors.HexColor('#333333'))
ref_style     = S('Ref',    fontSize=8.5,leading=12, spaceAfter=2)

def H1(t): return Paragraph(t, h1_style)
def H2(t): return Paragraph(t, h2_style)
def H3(t): return Paragraph(t, h3_style)
def P(t):  return Paragraph(t, body_style)
def B(t):  return Paragraph(f'• {t}', bullet_style)
def Cap(t):return Paragraph(t, cap_style)
def SP(n=6):return Spacer(1, n)
def HR():  return HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#cccccc'), spaceAfter=4, spaceBefore=4)

def fig(fname, caption, w_cm=14):
    path = FIG_DIR / fname
    if not path.exists():
        return [P(f'[Figure not found: {fname}]')]
    return [Image(str(path), width=w_cm*cm, height=w_cm*cm*0.55), Cap(caption), SP(4)]

def tbl(data, colWidths, header_bg=colors.HexColor('#2c5282')):
    t = Table(data, colWidths=colWidths)
    t.setStyle(TableStyle([
        ('BACKGROUND',     (0,0), (-1,0), header_bg),
        ('TEXTCOLOR',      (0,0), (-1,0), colors.white),
        ('FONTNAME',       (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',       (0,0), (-1,-1), 8.5),
        ('ALIGN',          (1,0), (-1,-1), 'CENTER'),
        ('ALIGN',          (0,0), (0,-1), 'LEFT'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f0f4fa')]),
        ('GRID',           (0,0), (-1,-1), 0.4, colors.HexColor('#bbbbbb')),
        ('TOPPADDING',     (0,0), (-1,-1), 3),
        ('BOTTOMPADDING',  (0,0), (-1,-1), 3),
        ('LEFTPADDING',    (0,0), (-1,-1), 5),
    ]))
    return t

NAVY = colors.HexColor('#1B2A4A')
story = []

# ── TITLE ──
story += [
    SP(10),
    Paragraph('Robust Time-Series Forecasting Under Noise and Anomaly Injection:<br/>A Comparative Study of ARIMA, RNN, LSTM, and Temporal Transformer', title_style),
    SP(6),
    Paragraph('Toprak Zeybek', author_style),
    Paragraph('Student ID: 323011022', inst_style),
    Paragraph('SEDS 537 Machine Learning  |  Izmir Institute of Technology  |  Spring 2026', inst_style),
    Paragraph('Instructor: Prof. Dr. Aytug Onan', inst_style),
    SP(12),
    HR(),
]

# ── ABSTRACT ──
story += [
    Paragraph('<b>Abstract.</b>', h2_style),
    Paragraph(
        'Real-world time-series data are rarely clean; sensor streams routinely exhibit Gaussian '
        'measurement noise and sudden point anomalies. This paper conducts a systematic robustness study '
        'on the Jena Climate dataset, training ARIMA, Vanilla RNN, Stacked LSTM, and a Temporal Transformer '
        'entirely on clean data, then evaluating each model under five levels of additive Gaussian noise '
        '(sigma in {0, 0.25, 0.5, 1.0, 2.0}) and five point-anomaly injection ratios (r in {0, 0.01, 0.05, 0.10, 0.20}). '
        'On clean data all neural models reach MAE ~0.06, outperforming ARIMA (MAE = 0.877) by an order of magnitude. '
        'Under Gaussian corruption the Transformer degrades most gracefully at low noise (sigma <= 0.5), consistent with '
        'its global self-attention hypothesis, but at high noise (sigma = 2.0) the LSTM achieves lower MAE (0.531 vs. 0.604). '
        'MC-Dropout uncertainty rises monotonically with corruption severity, confirming its value as a data-quality signal.',
        abstract_style,
    ),
    Paragraph('<b>Keywords:</b> Time-series forecasting, Robustness, Transformer, LSTM, Gaussian noise, Point anomalies, MC-Dropout', kw_style),
    HR(),
]

# ── 1. INTRODUCTION ──
story += [H1('1  Introduction')]
story += [
    P('Accurate time-series forecasting is foundational to climate monitoring, energy management, industrial '
      'process control, and financial modelling. Deep learning approaches -- especially LSTMs and Transformers -- '
      'consistently outperform classical statistical methods on standard benchmarks. However, a critical but '
      'underexplored dimension is <b>robustness</b>: how well does a model maintain accuracy when the test-time '
      'input is corrupted by noise or anomalies?'),
    P('The Temporal Transformer is theoretically motivated to be more robust than sequential models. Its '
      'multi-head self-attention computes a weighted aggregate over <i>all</i> positions in the input window '
      'simultaneously. A single corrupted time-step can therefore be downweighted and "outvoted" by the clean '
      'majority. In contrast, RNN and LSTM propagate a hidden state forward sequentially -- once a corrupted '
      'step contaminates the hidden state, that corruption is carried forward.'),
    P('This paper tests that hypothesis empirically with the following contributions: (1) a systematic robustness '
      'benchmark comparing all four model classes under 10 corruption conditions; (2) MC-Dropout as a '
      'corruption-aware uncertainty signal; (3) a detailed failure-case and ablation analysis; (4) open-source '
      'reproducible code, available at: ' + GITHUB_URL),
]

# ── 2. RELATED WORK ──
story += [H1('2  Related Work')]
story += [
    H3('Classical forecasting.'),
    P('ARIMA [Box & Jenkins 1970] and SARIMA remain widely used univariate baselines. Their autoregressive '
      'formulation makes them inherently robust to test-time input corruption: forecasts are generated purely '
      'from fitted model parameters, not from observed test windows.'),
    H3('Deep learning for time series.'),
    P('LSTMs [Hochreiter & Schmidhuber 1997] address vanishing gradients through gating mechanisms. '
      'Stacked LSTM architectures with dropout regularisation have achieved state-of-the-art performance '
      'on numerous temporal benchmarks.'),
    H3('Transformers for time series.'),
    P('Since the Transformer [Vaswani et al. 2017], several works have adapted self-attention to forecasting: '
      'Informer [Zhou et al. 2021] for long sequences, Temporal Fusion Transformers [Lim et al. 2021] for '
      'interpretable multi-horizon forecasting, and FEDformer [Zhou et al. 2022] for frequency-domain representations.'),
    H3('Robustness studies.'),
    P('Robustness to corrupted inputs is well-studied for image classification [Hendrycks & Dietterich 2019], '
      'but systematic evaluations for time-series remain rare. To our knowledge, no prior work performs the '
      'joint Gaussian noise + point anomaly robustness sweep presented here.'),
]

# ── 3. METHODOLOGY ──
story += [H1('3  Methodology')]
story += [
    H2('3.1  Data Preprocessing'),
    P('The <b>Jena Climate dataset</b> contains 14 atmospheric variables recorded every 10 minutes '
      '(2009-2016, 420,551 rows). We subsample to hourly resolution (every 6th row, 70,091 rows) and '
      'retain 13 numerical features. The target is air temperature T (deg C). Features are standardised '
      'with z-score normalisation (mean and std from training set only -- no data leakage). '
      'Split: 70% train / 15% val / 15% test. '
      'Sliding windows of W = 120 hourly steps (5 days), horizon H = 1.'),
    P('<b>Dataset availability:</b> The Jena Climate dataset is publicly available at:<br/>' + DATASET_URL),
    H2('3.2  Model Architectures'),
    H3('ARIMA (baseline 1).'),
    P('ARIMA(5,1,0) fitted on last 2,000 training observations. Does not consume test-time windows -- '
      'trivially robust to input corruption but cannot exploit multivariate context.'),
    H3('Vanilla RNN (baseline 2).'),
    P('Single-layer RNN, 64 hidden units. Final hidden state projected to scalar. ~7,000 parameters.'),
    H3('Stacked LSTM (baseline 3).'),
    P('Two LSTM layers, 128 hidden units, dropout = 0.2. MC-Dropout uncertainty: 50 forward passes at '
      'inference with dropout active, yielding predictive mean and std. ~200,000 parameters.'),
    H3('Temporal Transformer (proposed method).'),
    P('Linear projection (13 -> 64) + sinusoidal positional encoding + 2 Transformer encoder layers '
      '(4 heads, FFN dim = 256, dropout = 0.1) + linear head on last token. ~400,000 parameters. '
      'Motivation: self-attention aggregates all positions simultaneously, diluting the effect of localised corruptions.'),
    H2('3.3  Corruption Protocols'),
    P('<b>Additive Gaussian noise:</b> x_tilde = x + epsilon, epsilon ~ N(0, sigma^2), '
      'sigma in {0, 0.25, 0.5, 1.0, 2.0}. At sigma = 1.0 the noise std equals the training data std.'),
    P('<b>Point anomaly injection:</b> Fraction r of positions replaced with N(0, 25) outliers, '
      'r in {0, 0.01, 0.05, 0.10, 0.20}. All models trained on clean data only.'),
]

# ── 4. EXPERIMENTAL SETUP ──
story += [H1('4  Experimental Setup')]
story += [P(
    'All neural models trained for 5 epochs, batch size 256, Adam (lr = 1e-3), gradient clipping (max norm = 1.0), '
    'ReduceLROnPlateau scheduler (patience = 2, factor = 0.5). Hardware: MacBook Pro, Apple M-series, 17 GB RAM, PyTorch 2.x.'
)]
hp_data = [
    ['Parameter', 'Value', 'Model(s)'],
    ['Input window W',    '120 hourly steps',  'RNN, LSTM, Transformer'],
    ['Forecast horizon H','1',                 'All'],
    ['Batch size',        '256',               'Neural models'],
    ['Epochs',            '5',                 'Neural models'],
    ['Learning rate',     '1e-3',              'Neural models'],
    ['RNN hidden units',  '64',                'RNN'],
    ['LSTM hidden units', '128 x 2 layers',    'LSTM'],
    ['LSTM dropout',      '0.2',               'LSTM'],
    ['MC-Dropout samples','50',                'LSTM'],
    ['d_model',           '64',                'Transformer'],
    ['Attention heads',   '4',                 'Transformer'],
    ['Encoder layers',    '2',                 'Transformer'],
    ['ARIMA order',       '(5, 1, 0)',          'ARIMA'],
]
story += [SP(4), tbl(hp_data, [5.5*cm, 4.5*cm, 5*cm]), Cap('Table 1. Hyperparameter summary.'), SP(6)]

# ── 5. RESULTS ──
story += [H1('5  Results')]
story += [H2('5.1  Clean Test Set Performance')]
story += [P('Table 2 shows baseline performance on clean data. All neural models outperform ARIMA by ~14x in MAE. '
            'LSTM leads on MAE (0.0592), Transformer leads on RMSE (0.0824) and MAPE (30.80%).')]
clean_data = [
    ['Model', 'MAE', 'RMSE', 'MAPE (%)'],
    ['ARIMA',                '0.8769', '1.0411', '350.87'],
    ['Vanilla RNN',          '0.0602', '0.0860', '34.05'],
    ['Stacked LSTM',         '0.0592', '0.0829', '31.32'],
    ['Temporal Transformer', '0.0599', '0.0824', '30.80'],
]
story += [SP(4), tbl(clean_data, [5.5*cm, 3*cm, 3*cm, 3.5*cm]), Cap('Table 2. Clean test set performance.'), SP(6)]

story += [H2('5.2  Robustness to Gaussian Noise')]
story += [P('Table 3 shows MAE under increasing Gaussian noise. ARIMA is flat (input-independent). '
            'RNN degrades fastest (+1,183% at sigma=2.0). Transformer is most robust at sigma <= 0.5. '
            'At sigma >= 1.0, LSTM overtakes Transformer -- LSTM gating provides better filtering at extreme noise.')]
gauss_data = [
    ['Model',                'sigma=0.00','sigma=0.25','sigma=0.50','sigma=1.00','sigma=2.00'],
    ['ARIMA',                '0.8769','0.8769','0.8769','0.8769','0.8769'],
    ['Vanilla RNN',          '0.0602','0.1408','0.2516','0.4630','0.7723'],
    ['Stacked LSTM',         '0.0592','0.1078','0.1811','0.3223','0.5313'],
    ['Temporal Transformer', '0.0599','0.0971','0.1600','0.3079','0.6035'],
]
story += [SP(4), tbl(gauss_data, [4.5*cm, 2.3*cm, 2.3*cm, 2.3*cm, 2.3*cm, 2.3*cm]),
          Cap('Table 3. MAE under Gaussian noise corruption.'), SP(6)]
story += fig('mae_vs_gaussian_noise.png', 'Fig. 1. MAE vs. Gaussian noise level sigma for all models.')

story += [H2('5.3  Robustness to Point Anomalies')]
story += [P('Table 4 shows MAE under point anomaly injection. Transformer leads at r <= 5%. '
            'At r >= 10%, LSTM becomes competitive or better. RNN degrades most severely.')]
anom_data = [
    ['Model',                'r=0.00','r=0.01','r=0.05','r=0.10','r=0.20'],
    ['ARIMA',                '0.8769','0.8769','0.8769','0.8769','0.8769'],
    ['Vanilla RNN',          '0.0602','0.1692','0.4386','0.6118','0.8219'],
    ['Stacked LSTM',         '0.0592','0.1220','0.3000','0.4538','0.6894'],
    ['Temporal Transformer', '0.0599','0.1116','0.2865','0.4638','0.7322'],
]
story += [SP(4), tbl(anom_data, [4.5*cm, 2.3*cm, 2.3*cm, 2.3*cm, 2.3*cm, 2.3*cm]),
          Cap('Table 4. MAE under point anomaly injection.'), SP(6)]
story += fig('mae_vs_anomaly_ratio.png', 'Fig. 2. MAE vs. point anomaly injection ratio for all models.')

story += [H2('5.4  MC-Dropout Uncertainty')]
story += [P('LSTM MC-Dropout predictive std increases monotonically with corruption (Table 5): '
            '0.0305 (clean) -> 0.0868 (sigma=2.0) and 0.0799 (r=20%). '
            'MC-Dropout can serve as a lightweight data-quality detector at deployment time.')]
mc_data = [
    ['Corruption type',   'Level 0','Level 1','Level 2','Level 3','Level 4'],
    ['Gaussian (sigma)',  '0.0305', '0.0345', '0.0424', '0.0611', '0.0868'],
    ['Anomaly ratio (r)', '0.0308', '0.0365', '0.0564', '0.0671', '0.0799'],
]
story += [SP(4), tbl(mc_data, [4.5*cm, 2.3*cm, 2.3*cm, 2.3*cm, 2.3*cm, 2.3*cm]),
          Cap('Table 5. LSTM MC-Dropout mean predictive std vs. corruption level.'), SP(6)]
story += fig('mc_dropout_bands.png', 'Fig. 3. LSTM MC-Dropout uncertainty bands on clean test set.')
story += fig('robustness_degradation_gaussian.png', 'Fig. 4. Relative MAE increase (%) vs. clean baseline.')

# ── 6. ABLATION STUDY ──
story += [H1('6  Ablation Study')]
story += [P('To identify which Transformer components contribute to accuracy and robustness, five variants '
            'are trained for 3 epochs on 20,000 training samples.')]
abl_data = [
    ['Variant',           'Clean', 'sigma=0.5', 'sigma=1.0', 'Anomaly 5%'],
    ['Full (proposed)',   '0.0715','0.1599','0.2916','0.2877'],
    ['No positional enc.','0.0927','0.1932','0.3278','0.2693'],
    ['1 encoder layer',   '0.0726','0.1724','0.3028','0.2897'],
    ['d_model = 32',      '0.0818','0.1589','0.2793','0.2592'],
    ['No dropout',        '0.0618','0.1734','0.3158','0.2673'],
]
story += [SP(4), tbl(abl_data, [5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 3*cm]),
          Cap('Table 6. Ablation study -- MAE under clean and corrupted conditions.'), SP(6)]
story += [
    B('<b>Positional encoding is the most critical component:</b> removing it raises clean MAE by 30% (0.0715 -> 0.0927).'),
    B('<b>Depth matters modestly:</b> 1 layer vs. 2 increases clean MAE by 1.5%; greater impact under corruption.'),
    B('<b>d_model=32 is competitive:</b> matches or outperforms full model under corruption -- smaller models overfit less.'),
    B('<b>Dropout trades clean accuracy for robustness:</b> no-dropout variant achieves best clean MAE (0.0618) '
      'but degrades more under noise (0.3158 vs. 0.2916).'),
    SP(6),
]

# ── 7. ERROR ANALYSIS ──
story += [H1('7  Error Analysis')]
story += [
    H2('7.1  Error Distributions'),
    P('Absolute error distributions are right-skewed with a sharp peak near zero. '
      'Median |e| ~0.043; 90th-percentile errors: RNN 0.1315, LSTM 0.1296, Transformer 0.1264.'),
    H2('7.2  Temporal Error Pattern'),
    P('MAE binned by position in the test set (20 bins) is broadly stable, with a modest 5-10% '
      'increase toward the end, possibly reflecting mild seasonality mismatch.'),
    H2('7.3  Clean vs. Heavily Corrupted'),
    P('Under sigma=2.0: RNN x12.8, LSTM x8.9, Transformer x10.1. '
      'LSTM gating is the most resilient architecture under extreme noise.'),
]
story += fig('error_distributions.png', 'Fig. 5. Absolute error distributions on clean test set.')
story += fig('clean_vs_corrupted_mae.png', 'Fig. 6. MAE: clean vs. heavily corrupted (sigma=2.0).')

# ── 8. DISCUSSION ──
story += [H1('8  Discussion')]
story += [
    H3('Does the attention-dilution hypothesis hold?'),
    P('Partially. At low-to-moderate corruption (sigma <= 0.5, r <= 5%), the Transformer is consistently '
      'the most robust neural model. This advantage erodes at high corruption where the LSTM becomes superior. '
      'Possible explanations: (1) at very high noise, attention scores become nearly uniform, eliminating the '
      'dilution benefit; (2) LSTM forget gates may be more selective in filtering extreme outliers.'),
    H3('ARIMA as a robustness reference.'),
    P('ARIMA\'s flat curve illustrates the fundamental trade-off: models exploiting rich input context '
      'perform better on clean data but become vulnerable when that context is corrupted.'),
    H3('MC-Dropout as a data-quality monitor.'),
    P('The monotone increase in predictive std with corruption suggests MC-Dropout can be deployed '
      'as a lightweight online data-quality detector requiring no labelled anomaly data.'),
    H3('Limitations.'),
    P('(1) 5 epochs only. (2) Small Transformer (d_model=64, 2 layers). '
      '(3) Single-step forecasting (H=1). (4) Synthetic corruption may differ from real sensor faults.'),
]

# ── 9. CONCLUSION ──
story += [H1('9  Conclusion')]
story += [
    P('We presented a systematic robustness study comparing ARIMA, Vanilla RNN, Stacked LSTM, and a '
      'Temporal Transformer on the Jena Climate dataset. Key findings:'),
    B('All neural models achieve MAE ~0.06 on clean data, outperforming ARIMA (MAE = 0.877) by ~14x.'),
    B('Temporal Transformer is most robust under low-to-moderate corruption, supporting the attention-dilution hypothesis.'),
    B('At high corruption (sigma >= 1.0 or anomaly ratio >= 10%), Stacked LSTM achieves lower MAE.'),
    B('MC-Dropout uncertainty increases monotonically with corruption -- useful as a data-quality indicator.'),
    B('Positional encoding is the most critical Transformer component; dropout improves robustness.'),
    SP(4),
    P('Future work: attention-based anomaly masking, adversarial training, multi-step forecasting horizons.'),
]

# ── REFERENCES ──
story += [H1('References')]
refs = [
    '[1] Vaswani, A. et al. (2017). Attention is all you need. NeurIPS.',
    '[2] Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. Neural Computation 9(8).',
    '[3] Box, G.E.P. & Jenkins, G.M. (1970). Time Series Analysis: Forecasting and Control. Holden-Day.',
    '[4] Lim, B. et al. (2021). Temporal fusion transformers. Int. J. Forecasting 37(4).',
    '[5] Zhou, H. et al. (2021). Informer: Beyond efficient transformer. AAAI.',
    '[6] Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. ICML.',
    '[7] Hendrycks, D. & Dietterich, T. (2019). Benchmarking neural network robustness. ICLR.',
    '[8] Hyndman, R.J. & Athanasopoulos, G. (2018). Forecasting: Principles and Practice. OTexts.',
    '[9] Greff, K. et al. (2017). LSTM: A search space odyssey. IEEE TNNLS 28(10).',
    '[10] Wu, H. et al. (2021). Autoformer: Decomposition transformers. NeurIPS.',
    '[11] Zhou, T. et al. (2022). FEDformer: Frequency enhanced decomposed transformer. ICML.',
    '[12] Zhu, L. & Laptev, N. (2017). Deep and confident prediction for time series at Uber. ICDMW.',
]
for r in refs:
    story.append(Paragraph(r, ref_style))

# ── BUILD ──
doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=2.5*cm, rightMargin=2.5*cm,
    topMargin=2.5*cm,  bottomMargin=2.5*cm,
    title='Robust Time-Series Forecasting',
    author='Toprak Zeybek',
)
doc.build(story)
print('Done -> ' + str(OUT))
