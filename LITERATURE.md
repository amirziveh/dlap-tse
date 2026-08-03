# Related Literature — Deep Learning in Asset Pricing on the Tehran Stock Exchange

**Prepared:** 2026-07-31 · **Source paper:** Chen, Pelger & Zhu, "Deep Learning in Asset Pricing," *Management Science* 70(2), 714–750 (Feb 2024; online July 2023). DOI 10.1287/mnsc.2023.4695

**Verification note:** International entries were verified against Crossref, publisher pages, Semantic Scholar, and arXiv. Iranian entries were verified against NoorMags.ir (meta descriptions), SID.ir, and journal pages. Items that could not be verified are explicitly flagged. Persian years are given in Shamsi (SH) with Gregorian conversion; a 1403 SH paper published in winter ≈ 2025 Gregorian — confirm exact dates when citing.

---

## PART 1. INTERNATIONAL LITERATURE (outside Iran)

### 1.1 The ML-in-asset-pricing canon (benchmarks and foundations)

| # | Citation | Contribution / relevance |
|---|---|---|
| 1 | Gu, Kelly & Xiu (2020). "Empirical Asset Pricing via Machine Learning." *Review of Financial Studies* 33(5), 2223–2273 | The foundational ML cross-section benchmark: penalized linear, tree, and neural models from firm characteristics. Primary performance benchmark for the TSE replication. |
| 2 | Kelly, Pruitt & Su (2019). "Characteristics Are Covariances: A Unified Model of Risk and Return." *JFE* 134(3), 501–524 | Instrumented PCA — latent factors with characteristics-dependent loadings; direct antecedent of CPZ's nonlinear factor structure. |
| 3 | Kozak, Nagel & Santosh (2020). "Shrinking the Cross-Section." *JFE* 135(2), 271–292 | Sparse/penalized SDF from many characteristic portfolios; the shrinkage benchmark CPZ compares against. |
| 4 | Freyberger, Neuhierl & Weber (2020). "Dissecting Characteristics Nonparametrically." *RFS* 33(5), 2326–2377 | Group-LASSO selection of which characteristics matter; benchmark for variable selection in sparse markets like TSE. |
| 5 | Feng, Giglio & Xiu (2020). "Taming the Factor Zoo: A Test of New Factors." *JF* 75(3), 1327–1370 | Double-selection LASSO incremental-value test; standard evaluation of new factors. |
| 6 | Harvey & Liu (2021). "Lucky Factors." *JFE* 141(2), 413–435 | Multiple-testing corrections for factor discovery; motivates the factor-zoo problem CPZ's SDF compression addresses. |

### 1.2 SDF estimation (theory and method)

| # | Citation | Contribution / relevance |
|---|---|---|
| 7 | Hansen & Jagannathan (1991). "Implications of Security Market Data for Models of Dynamic Economies." *JPE* 99(2), 225–262 | HJ bound and HJ distance — the theoretical basis for evaluating SDFs by maximal Sharpe ratio, which CPZ's losses directly target. |
| 8 | Cochrane (2011). "Presidential Address: Discount Rates." *JF* 66(4), 1047–1108 | The discount-rate view and the case for estimating the SDF rather than only expected returns; conceptual foundation of CPZ. |
| 9 | Lettau & Pelger (2020). "Estimating Latent Asset-Pricing Factors." *J. Econometrics* 218(1), 1–31 | R-PCA: PCA with a pricing-error penalty; the direct methodological antecedent of CPZ's penalized SDF loss. |
| 10 | Bryzgalova, Pelger & Zhu (2025). "Forest Through the Trees: Building Cross-Sections of Stock Returns." *JF* 80(5), 2447–2506 (online Sept 2025; 2023 dates refer to the SSRN working paper 3493458) | Tree-based sparse cross-sections spanning the SDF; companion paper sharing CPZ's SDF-spanning philosophy. |
| 11 | Nagel (2021). *Machine Learning in Asset Pricing.* Princeton University Press | Book synthesizing ML cross-sectional pricing from the SDF perspective; key reference for the theory section. |
| 12 | Giglio & Xiu (2021). "Asset Pricing with Omitted Factors." *JPE* 129(7), 1947–1990 | SDF-based alpha/risk-premium estimates robust to omitted factors; supports SDF-based evaluation over factor lists. |
| 13 | Aleti & Bollerslev (2025). "News and Asset Pricing: A High-Frequency Anatomy of the SDF." *RFS* 38(3), 712–759 | Recent SDF estimation over news/high-frequency information. |
| — | ⚠️ *Unverified:* "Through the Looking Glass: Diversified SDFs" (Bryzgalova-Pelger-Zhu) and "Machine Learning the SDF: Simple, Robust and Fast" could not be located in Crossref/Semantic Scholar/arXiv. Do not cite without checking SSRN directly. |

### 1.3 Autoencoder / deep-learning extensions of CPZ

| # | Citation | Contribution / relevance |
|---|---|---|
| 14 | Gu, Kelly & Xiu (2021). "Autoencoder Asset Pricing Models." *J. Econometrics* 222(1), 429–450 | Deep-learning latent factors with a no-arbitrage restriction; the most direct DL competitor to CPZ — natural second model for our replication. |
| 15 | Feng, He, Polson & Xu (2024). "Deep Learning in Characteristics-Sorted Factor Models." *JFQA* 59(7), 3001–3036 | Deep factor models with economic objective (PPN/CRP losses); explicit CPZ extension; useful for comparing estimation losses on TSE. |
| 16 | Chen, Wang & Huang (2026). "Variational Autoencoder Asset Pricing Models with Economic Restrictions." *Int. Review of Economics & Finance* 109, 105402 | VAE variant of the CPZ/GKX autoencoder family. |
| 17 | Wang, Cheng & Wang (2025). "NewsNet-SDF: Stochastic Discount Factor Estimation with Pretrained Language Model News Embeddings via Adversarial Networks." arXiv:2505.06864 | Adversarial-SDF estimation (CPZ's method) + LLM news embeddings; clearest recent example of CPZ's loss being built upon. |
| 18 | Guijarro-Ordonez, Pelger & Zanotti (2025). "Deep Learning Statistical Arbitrage." *Management Science* (online Dec 2025) | Pelger's own deep-learning extension; illustrates the CPZ research program. |
| 19 | Dixon, Polson & Goicoechea (2022). "Deep Partial Least Squares for Empirical Asset Pricing." SSRN 4137647 (working paper) | Deep nonlinear factor structure for individual returns; same family. |

### 1.4 Macro-state conditional models (CPZ's conditioning information)

| # | Citation | Contribution / relevance |
|---|---|---|
| 20 | Ludvigson & Ng (2007). "The Empirical Risk–Return Relation: A Factor Analysis Approach." *JFE* 83(1), 171–222 | Macro-factor extraction and time-varying risk premia — the economic motivation for CPZ's conditional (macro-lagged) models. |
| 21 | Ludvigson & Ng (2009). "Macro Factors in Bond Risk Premia." *RFS* 22(12), 5027–5067 | Macro factors as conditioning information; template for CPZ's macro-state variables. |
| 22 | Lettau & Ludvigson (2001). "Consumption, Aggregate Wealth, and Expected Stock Returns." *JF* 56(3), 815–849 | The *cay* variable — a key candidate conditioning variable for a TSE replication. |
| 23 | Brandt, Santa-Clara & Valkanov (2009). "Parametric Portfolio Policies: Exploiting Characteristics in the Cross-Section of Equity Returns." *RFS* 22(9), 3411–3447 | Directly parameterizing portfolio weights in characteristics (utility-based); the alternative view and antecedent of deep portfolio policies. |

### 1.5 International evidence and emerging markets (our paper's positioning)

| # | Citation | Contribution / relevance |
|---|---|---|
| 24 | Tobek & Hronec (2021). "Does It Pay to Follow Anomalies Research? Machine Learning Approach with International Evidence." *J. Financial Markets* 56, 100588 ⚠️ *(JFM, not JFE as sometimes cited)* | ML anomaly strategies outside the US; evidence that US anomaly sets transfer imperfectly — directly relevant to applying CPZ outside the US. |
| 25 | Han, He, Rapach & Zhou (2024). "Cross-Sectional Expected Returns: New Fama–MacBeth Regressions in the Era of Machine Learning." *Review of Finance* 28(6), 1807–1831 ⚠️ *(the team's "across the Globe" title could not be verified; this is the verified publication)* | ML in Fama–MacBeth regressions with global evidence. |
| 26 | Leippold, Wang & Zhou (2022). "Machine Learning in the Chinese Stock Market." *JFE* 145(2), 64–82 | GKX-style ML on China; the most-cited EM application and best template for our model comparisons. |
| 27 | Hanauer & Kalsbach (2023). "Machine Learning and the Cross-Section of Emerging Market Stock Returns." *Emerging Markets Review* 55, 101022 | ML cross-section on broad EM panels; closest direct precedent for an EM replication of the ML canon, including DL. |
| 28 | Hanauer & Lauterbach (2019). "The Cross-Section of Emerging Market Stock Returns." *Emerging Markets Review* 38, 265–286 | Which US anomalies work in EMs; baseline evidence our TSE results should be compared against. |

### 1.6 Characteristics data

| # | Citation | Contribution / relevance |
|---|---|---|
| 29 | Chen & Zimmermann (2022). "Open Source Cross-Sectional Asset Pricing." *Critical Finance Review* 11(2), 207–264 | The OSAP database of 150+ anomaly characteristics with open-source implementation; the reference for characteristic construction (CPZ use its ~40-characteristic set). |

### 1.7 Recent additions (2023–2026) and surveys

| # | Citation | Contribution / relevance |
|---|---|---|
| 30 | Giglio, Kelly & Xiu (2022). "Factor Models, Machine Learning, and Asset Pricing." *Annual Review of Financial Economics* 14(1), 337–368 | Authoritative survey linking factor models, ML, and SDF estimation. |
| 31 | Kelly & Xiu (2023). "Financial Machine Learning." *Foundations and Trends in Finance* 13(3–4), 205–363 | Most comprehensive recent survey of ML in finance, incl. deep learning SDFs. |
| 32 | Avramov, Cheng & Metzker (2023). "Machine Learning vs. Economic Restrictions: Evidence from Stock Return Predictability." *Management Science* 69(5), 2587–2619 | ML with economic constraints; informs constrained vs unconstrained SDF comparison. |
| 33 | Avramov, Cheng, Metzker & Voigt (2023). "Integrating Factor Models." *JF* 78(3), 1593–1646 | Integrates time-series and cross-sectional factor models via ML. |
| 34 | Kelly, Malamud & Zhou (2024). "The Virtue of Complexity in Return Prediction." *JF* 79(1), 459–503 | Theoretical result that complex (deep) models dominate simple ones for pricing kernels — directly supportive of CPZ. |
| 35 | Feng, Lan, Wang & Zhang (2026). "Selecting and Testing Asset-Pricing Models: A Stepwise Approach." *Management Science* (online June 2026) | New factor-model selection and testing method. |
| 36 | Simon, Weibels & Zimmermann (2026). "Deep Parametric Portfolio Policies." *Management Science* (online June 2026) | Deep-learning version of BSCV portfolio policies; the utility-side counterpart to CPZ's SDF-side deep models. |
| 37 | Kolm & Ritter (2025). "Reinforcement Learning for Asset and Portfolio Management." *J. Portfolio Management* 52(2), 81–95 | Survey of RL in asset management. |
| 38 | Lopez-Lira & Tang (2023). "Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models." arXiv:2304.07619; **forthcoming JFE** | The flagship LLM return-predictability paper; anchor of the LLM branch. |
| 39 | Bollerslev, Li & Tang (2026). "Forecasting and Managing Correlation Risks." *Management Science* (online April 2026) | Recent ML risk-management paper in CPZ's citing network (peripheral). |
| 40 | Weigand (2019). "Machine Learning in Empirical Asset Pricing." *Financial Markets and Portfolio Management* 33(1), 93–104 | Early survey; historical framing only. |

### 1.8 Source paper details & official code (verified)

- **Paper:** Chen, L., Pelger, M. & Zhu, J. (2024). "Deep Learning in Asset Pricing." *Management Science* 70(2), 714–750. DOI 10.1287/mnsc.2023.4695. arXiv:1904.00745 (2019 working paper). SSRN 3350138. ⚠️ *PLAN.md currently cites 70(1)/2023 — correct to 70(2)/2024.*
- **Official code (verified accessible):**
  - `github.com/LouisChen1992/Deep_Learning_Asset_Pricing` (41★) — full training code: `src/`, `config/config.json`, `run.py`, RF extensions (`config_RF`, `create_RF_data.py`). **TensorFlow 1.12.0 / Python 3.6.**
  - `github.com/LouisChen1992/Deep_Learning_in_Asset_Pricing` (152★) — results notebooks: `model_GAN.ipynb` (adversarial SDF), `model_FFN.ipynb`, `model_Linear.ipynb`; results tables (Sharpe: GAN 0.75 test vs EN 0.50; EV: GAN 0.08; XS-R²: GAN 0.23).
  - **Data:** Google Drive (`Char_train.npz` characteristics 46-dim, `macro_train.npz` macro 178-dim, split train/valid/test); README's earlier Dropbox `Data.xlsx`.
  - **Architecture (config.json):** LSTM state extractor (4 units) → 2×64 FFN; dropout 0.95; Adam lr 1e-3; windows 240m/60m/300m; weighted loss.
  - **Implication for replication:** official code is TF 1.12 (legacy). PLAN.md's PyTorch choice is sensible — port the architecture, keep the evaluation protocol.

---

## PART 2. IRANIAN LITERATURE (inside Iran)

### 2.1 English-language papers on TSE (ML/DL prediction)

| # | Citation | Contribution |
|---|---|---|
| 41 | Ebrahimpour, Nikoo, Masoudnia & Yousefi (2011). "Mixture of MLP-experts for trend forecasting of time series." *Int. J. Forecasting* 27(4) | Mixture-of-experts MLP for TSE index trend. |
| 42 | Hoseinzade & Haratizadeh (2019). "CNNpred: CNN-based stock market prediction using a diverse set of variables." *Expert Systems with Applications* 129 | CNN with 2D input matrices (technical+macro); benchmarked on TSE among 4 markets. |
| 43 | Nabipour, Nayyeri, Jabani & Mosavi (2020). "Deep learning for stock market prediction." *Entropy* 22(8), 840 | LSTM/GRU/RNN/ANFIS vs SVM/RF/ANN on TSE sectors; LSTM best RMSE. |
| 44 | Hatefi Ghahfarrokhi & Shamsfard (2020). "Tehran stock exchange prediction using sentiment analysis of online textual opinions." *Intelligent Systems in Accounting, Finance and Management* 27(1) | Persian text sentiment + neural models for TSE index. |
| 45 | Aminimehr, Raoofi & Aminimehr (2020). "The role of feature engineering in prediction of TSE index based on LSTM." *Iranian Journal of …* ⚠️ journal title not fully verified | LSTM + feature engineering for TSE index. |
| 46 | Aminimehr, Bajalan & Hekmat (2021). "A study on the characteristics of TSE index return data and introducing a regime switching prediction method based on neural networks." *J. Financial Management Perspective* | Documents TSE index return properties (fat tails, regimes). |
| 47 | Dami & Esterabi (2021). "Predicting stock returns of Tehran exchange using LSTM neural network and feature engineering technique." *Multimedia Tools and Applications* 80 | LSTM + feature selection for TSE firm-level returns. |
| 48 | Azizi, Abdolvand & Ghalibaf Asl (2021). "The impact of Persian news on stock returns through text mining techniques." *Iranian J. Management Studies* 14(3) | Text-mining of Persian news → TSE returns. |
| 49 | Sohrabi, Rouhani, Yazdani, Khalili Jafarabad & Kazemi Movahed (2023). "Tehran Stock Exchange, stocks price prediction, using wisdom of crowd." *Iranian J. Finance* 7(4) | Social-media crowd features for TSE prediction. |
| 50 | Moradi, Jabbari Nooghabi & Rounaghi (2019). "Investigation of fractal market hypothesis and forecasting time series stock returns for TSE." *Int. J. Finance & Economics* 24(2) | Fractality + ANN/ARFIMA forecasting of TSE returns. |
| 51 | Ramezanian, Peymanfar & Ebrahimi (2019). "An integrated framework of genetic network programming and MLP for prediction of daily stock return." *Applied Soft Computing* 82 | GNP feature extraction + MLP for TSE daily return direction. |
| 52 | Jahangoshai Rezaee, Jozmaleki & Valipour (2018). "Integrating dynamic fuzzy C-means, data envelopment analysis and ANN…" *Physica A* 506 | Fuzzy clustering + DEA + ANN pipeline for TSE firm performance. |
| 53 | Ghanbari (2019). "Machine learning application in stock price prediction: applied to the active firms in oil and gas industry in TSE." *Petroleum Business Review* 3(4) | ML for TSE oil/gas sector. |
| 54 | Esfahanipour & Mardani (2011). "An ANFIS model for stock price prediction: The case of Tehran stock exchange." INISTA/IEEE | ANFIS for TSE. |
| 55 | Bodaghi, Owhadi, Khalili Nasr & Khadem Sameni (2023). "A novel CNN-LSTM model for predicting the railway sector stock price in TSE." ICWR/IEEE | CNN-LSTM for TSE railway sector. |
| 56 | Tajik Gholami & Shahabi (2025). "Comparison of deep learning methods with traditional financial time series models for forecasting the TEPIX index." *Systems and Soft Computing* | DL (LSTM/GRU/RNN) vs ARIMA/GARCH on TEPIX. |
| 57 | Osoolian, Nikmaram & Karimi (2025). "Predicting index trend using hybrid neural networks with multi-scale temporal feature extraction in TSE." *Financial Research Journal* 27(4) | Hybrid CNN/LSTM multi-scale features for TSE index direction. |
| 58 | Saffarian & Haratizadeh (2024). "LLM-driven feature extraction for stock market prediction: a case study of TSE." IKT/IEEE | LLM features for TSE prediction. |
| 59 | Hashemipour, Zahedi & Fateh (2026). "A hybrid model for stock market prediction with stacked autoencoder for feature engineering." *Iranian J. Science and Technology* (Springer) | Stacked autoencoder features on TSE data. |
| 60 | Pooresmaeil Niaki, Peymany Foroushani & Amini (2025). "Predicting the trend of the total index of TSE using an image processing technique." *Iranian J. Finance* 9 | Candlestick-image CNN for TSE index. |
| 61 | Raei, Vahdati, Mohebbi et al. (2025). "Interpreting forecast of the return of the price index of manufacturing industries in TSE using explainable ensemble learning." *J. Financial Management Perspective* | Ensemble ML + SHAP for TSE industry returns. |
| 62 | Heidari Dalooei, Vahdati, Mohebbi & Bagherpour (2026). "Predictability of the TEDPIX using a combined machine learning approach." *Financial Research Journal* | XGBoost-GA, 84% directional accuracy 2019–2024, SHAP. |
| 63 | Mohebi & Mohebi (2026). "Predicting stock market returns using temporal fusion transformer." *Iranian J. Accounting, Auditing and Finance* | TFT transformer on TSE data. |
| 64 | Mehtari Taheri & Esfahanipour (2024). "CNN2D-MV: a hybrid model for stock portfolio trading system." SSRN (working paper) | 2D-CNN TSE portfolio trading. |
| 65 | Tavakoli & Doosti (2021). "Forecasting the Tehran stock market by machine learning methods using a new loss function." SID-hosted preprint ⚠️ | ML with custom loss; preprint only. |

### 2.2 English-language papers on TSE (asset pricing / SDF — directly relevant)

| # | Citation | Contribution |
|---|---|---|
| 66 | Davallou & Badri (2015). "Specific risk pricing: evidence from Fama-MacBeth model and stochastic discount factor (SDF)." *J. Financial Engineering and Securities Management* | Idiosyncratic risk priced on TSE via Fama-MacBeth and SDF/GMM — direct SDF-testing precedent. |
| 67 | Talakesh Na'ini, Taleblou, Mohammadi & Mohajeri (2023). "Evaluating the efficiency and robustness of beta and stochastic discount factor methods in Iranian stock market." *Iranian J. Economic Research* 27(90) | CAPM/FF3 in beta vs SDF form, GMM on 25 FF 5×5 TSE portfolios, 2000–2019. **Closest Iranian antecedent for test assets.** |
| 68 | Taleblou, Mohammadi, Morovvat & Bagheri Todeshki (2022). "Asset pricing modeling test based on behavioral stochastic discount factor (SDF): a case study of TSE." *J. Economic Studies and Policies* | Behavioral SDF (sentiment in utility) via Euler equations + GMM, 63 firms/18 industries. |
| 69 | Taleblou & Bagheri Todeshki (2024). "Sentiment as a risk factor in capital markets: an analysis of TSE within the SDF framework." *Iranian J. Economic Research* 29(98) | Turnover-based sentiment priced in SDF; behavioral SDF outperforms traditional on TSE. |
| 70 | Zare (2025). "Forecasting returns with a hybrid model: NNAR and CAPM for asset valuation." *J. Mathematics and Modeling in Finance* 5 | NNAR market-return forecast embedded in CAPM. |
| 71 | Fadaei (2026). "Deep sequential learning for asset return forecasting: an LSTM-enhanced capital asset pricing framework." *J. Mathematics and Modeling in Finance* 6 | LSTM market forecasts in CAPM (US sample). |
| 72 | Moradi, Nejat & Sardari Zarchi (2025). "Representing investment knowledge in terms of returns in the Iranian stock market using deep neural models under environmental uncertainty." *Financial Research Journal* | LSTM beats DQN/SVR/RF for weekly returns of 200 TSE firms. |

### 2.3 Persian-language papers — deep learning & neural networks on TSE

| # | Authors (SH year) | Title (Persian) | Journal |
|---|---|---|---|
| 73 | چاوشی & راعی (1382/2003) | پیشبینی بازده سهام در بورس اوراق بهادار تهران: مدل شبکههای عصبی مصنوعی و مدل چندعاملی | تحقیقات مالی |
| 74 | نمازی & کیامهر (1386/2007) | پیشبینی بازده روزانه سهام با شبکههای عصبی مصنوعی | تحقیقات مالی |
| 75 | راعی، محمدی & فندرسکی (1394/2015) | پیشبینی شاخص قیمت بورس با شبکه عصبی و تبدیل موجک | مدیریت دارایی و تأمین مالی |
| 76 | یمرعلی (1398/2019) | پیشبینی بازده غیرعادی سهام با رهیافت شبکههای عصبی | تحقیقات حسابداری و حسابرسی |
| 77 | شریففر، خلیلی عراقی، رئیسی وانانی & فلاح شمس (1401/2022) | کاربرد معماریهای یادگیری عمیق در پیشبینی قیمت سهام (CNN) | مدیریت دارایی و تأمین مالی 10(4) |
| 78 | سهرابی، میربرگکار، چیرانی & خردیار (1401/2022) | مدلسازی پیشبینی جهشهای شاخص بازار سهام با شبکه عصبی بازگشتی یادگیری عمیق | بورس اوراق بهادار 15(57) |
| 79 | کیانیزاده، باغانی & حمیدیان (1402/2023) | مقایسه دقت مدلهای منتخب یادگیری ماشین جهت پیشبینی قیمت سهام | بورس اوراق بهادار 16(61) |
| 80 | غلامی & شمسقارنه (1403/2024) | ارائه مدلی برای پیشبینی قیمت سهام مبتنی بر CNN-LSTM بهینهشده در TSE | چشمانداز مدیریت مالی 14(3) |
| 81 | صحرائی، قلیزاده، سپهردوست & حسینیدوست (1404/2025) | بهینیابی پرتفوی مبتنی بر پیشبینی؛ رویکرد NARX و LSTM | بورس اوراق بهادار 18(70) |
| 82 | گلتبار، ابونوری & حبیبنیا (1403/2025) | مدلسازی پیشبینی قیمت سهام در ایران با استفاده از یادگیری عمیق (LSAE) | تحقیقات مالی 27(3), 424–463 |
| 83 | حیدری & امیری (1401/2022) | بررسی قدرت مدلهای مبتنی بر هوش مصنوعی در پیشبینی روند قیمت سهام TSE | تحقیقات مالی 24(3) |
| 84 | محبی، فدائینژاد، اصولیان & حمیدیزاده (1401/2022) | انتخاب ویژگیهای مناسب برای مدل پیشبینی شاخص TSE بر مبنای کاهش ابعاد | تحقیقات مالی 24(4) |
| 85 | اصولیان، نیکمرام & کریمی (1404/2025) | پیشبینی روند شاخص کل با شبکههای عصبی هیبریدی (ویژگیهای چندمقیاسه) | تحقیقات مالی |
| 86 | حیدری دلوئی، وحدتی، محبی & باقرپور (1404/2026) | پیشبینیپذیری شاخص کل TSE با یادگیری ماشین ترکیبی (XGBoost-GA) | تحقیقات مالی |
| 87 | مرادی، نشاط & سرداری زارچی (1404/2025) | نمایش دانش سرمایهگذاری برحسب بازده با مدلهای عصبی عمیق | تحقیقات مالی |
| 88 | ترابیپور & سیادت (1401/2022) | روشی جهت پیشبینی قیمت سهام بازار بورس تهران مبتنی بر یادگیری عمیق (CNN-LSTM) | پدافند الکترونیکی و سایبری 11(1) |
| 89 | نمکی، شیرکوند & صفائیپور (1404/2025) | پیشبینی قیمت سهام با ترکیب RNN, LSTM و GRU و تجزیه فرکانسی | راهبرد مدیریت مالی 50 |
| 90 | حیدرزاده، صفا، فلاح شمس & جهانگیرنیا (1403/2024) | پیشبینیپذیری TSE با مدلهای یادگیری عمیق (CNN-LSTM هیبریدی) | مهندسی مدیریت نوین 35 |
| 91 | ذوالفقاری، سحابی & بختیاران (1399/2020) | پیشبینی بازده شاخص کل با مدلهای ترکیبی یادگیری عمیق و خانواده GARCH | مهندسی مالی و مدیریت اوراق بهادار 42 |
| 92 | عبداله زرکش & سیفیپور (1404/2026) | مدلسازی و پیشبینی نوسانات بازار سهام با RNN, LSTM و GRU | اقتصاد مالی 73 |
| 93 | تهرانی، فلاحپور & جعفری (1404/2025) | مدل ترکیبی پیشبینی جهت حرکت قیمت با LSTM، تکنیکال و اقتصاد کلان | مهندسی مالی و مدیریت اوراق بهادار 62 |
| 94 | عبدی، مرادزادهفرد، احمدزاده & خدام (1400/2022) | بهینهسازی سبد سهام بر اساس پیشبینی قیمت با LSTM | چشمانداز مدیریت مالی 36 |
| 95 | شمس & پارسائیان (1391/2012) | مقایسه عملکرد مدل فاما و فرنچ و شبکههای عصبی مصنوعی در پیشبینی بازده سهام | مهندسی مالی و مدیریت اوراق بهادار 11 |
| 96 | جعفری، میثاقی فاروجی & احمدوند (1392/2013) | مقایسه CAPM، سهعاملی فاما-فرنچ و شبکههای عصبی مصنوعی | پژوهشنامه اقتصاد و کسب و کار 5 |
| 97 | رئوفی & محمدی (1397/2018) | پیشبینی بازده بازار سهام تهران: موجک + شبکه عصبی فازی تطبیقی | پژوهشهای اقتصادی ایران 76 |
| 98 | هادیزاده، تارخ & میرزایی قزاآنی (1401/2023) | پیشبینی رفتار بورس با اندیکاتورهای تکنیکال: یادگیری تقویتی عمیق و CNN | پژوهشهای نوین در تصمیمگیری 7(4) |
| 99 | کاظمیان حسینآبادی، داودی، مشهدیزاده & جوزی (1403/2024) | سیستم معاملاتی الگوریتمیک بر پایه یادگیری تقویتی عمیق | دانش مالی تحلیل اوراق بهادار 61 |
| 100 | حاجی مولانا (1403/2024) | انتخاب پرتفوی با شبکه عصبی مصنوعی و یادگیری عمیق | مطالعات راهبردی مالی و بانکی 3 |
| 101 | آذر & کریمی (1388/2010) | پیشبینی بازده سهام با نسبتهای حسابداری و شبکههای عصبی | تحقیقات مالی 11(2) |
| 102 | باجلان، فلاحپور & دانا (1396/2017) | پیشبینی روند قیمت سهام با SVM تعدیلیافته + انتخاب ویژگی هیبرید | چشمانداز مدیریت مالی 7(4) |
| 103 | حافظی، شهرابی & هداوندی (1392/2013) | توسعه مدلی ترکیبی هوشمند برای پیشبینی بازار سهام تهران | تحقیق در عملیات در کاربردهای آن 10(2) |
| 104 | مرادزادهفرد، دارابی & شاهعلیزاده (1393/2014) | یکپارچهسازی تکنیکهای هوش مصنوعی جهت پیشبینی قیمت سهام | پژوهشهای حسابداری مالی و حسابرسی 6(22) |
| 105 | خدایاری، یعقوبنژاد & خلیلی عراقی (1399/2020) | مقایسه برآورد تلاطم بازارهای مالی با رگرسیون و شبکه عصبی | اقتصاد مالی 14(53) |
| 106 | ابراهیمی (1402/2023) | تحلیل پیشبینی در بازار سهام: درخت تصمیم vs ماشین بردار پشتیبان | حسابداری، امور مالی و هوش محاسباتی |
| 107 | اسماعیلی، عباسی & فلاح (1397/2018) | پیشبینی عملکرد کوتاهمدت IPO با kNN و SVM | چشمانداز مدیریت مالی 21 |
| 108 | جهانبازی (1404/2025) | پیشبینی شاخص کل بر اساس درآمدهای نفتی: LSTM، LSTM-Dense، رگرسیون | پژوهشهای نوین بینرشتهای |
| 109 | کیانیزاده (1402/2023) — see 79 | — | — |

### 2.4 Persian-language papers — SDF and asset pricing factors (directly relevant to CPZ)

| # | Authors (SH year) | Title (Persian) | Journal |
|---|---|---|---|
| 110 | طلاکش نایینی، طالبلو، محمدی & مهاجری (1401/2022) | ارزیابی کارایی و پایداری روشهای بتا و عامل تنزیل تصادفی در بازار سهام ایران | پژوهشهای اقتصادی ایران 27(90) |
| 111 | طالبلو، محمدی، مروت & باقری تودشکی (1401/2022) | آزمون الگوی قیمتگذاری دارایی بر اساس عامل تنزیل تصادفی (SDF) رفتاری | مطالعات و سیاستهای اقتصادی 9(2) |
| 112 | طالبلو & باقری تودشکی (1403/2024) | احساسات بهعنوان یک عامل ریسک در بازار سرمایه در چارچوب SDF | پژوهشهای اقتصادی ایران 29(98) |
| 113 | دولو & بدری (1393/2014) | قیمتگذاری ریسک خاص: شواهدی از فاما-مکبث و عامل تنزیل تصادفی | مهندسی مالی و مدیریت اوراق بهادار 5(20) |
| 114 | اشراقنیای جهرمی & نشوادیان (1387/2008) | آزمایش مدل سهعاملی فاما و فرنچ در بورس اوراق بهادار تهران | مهندسی صنایع و مدیریت شریف |
| 115 | عیوضلو، قهرمانی & عجم (1395/2016) | بررسی عملکرد مدل پنجعاملی فاما و فرنچ با استفاده از آزمون GRS | تحقیقات مالی 18(3) |
| 116 | نوربخش & ایرانی جانیارلو (1399/2020) | مقایسه FF3 با FF5 در پیشبینی بازده سهام TSE | دانش سرمایهگذاری 9(36) |
| 117 | داودی & صابراصفهانی (1399/2020) | بررسی کارایی مدل پنجعاملی فاما و فرنچ در پیشبینی بازده سهام | بورس اوراق بهادار 13(51) |
| 118 | نادری بنی، عربصالحی & کاظمی (1398/2019) | کشف ناهنجاری قیمتگذاری داراییها در سطح شرکت (بیزی سلسلهمراتبی) | حسابداری مالی 11(44) |
| 119 | سلیمانیان، فروغی & امیری (1398/2020) | بسط مدلهای عاملی قیمتگذاری از طریق عوامل ارزش، مومنتوم و کیفیت | حسابداری مالی 44 |
| 120 | راعی، بهاروند & موفق (1389/2011) | قیمتگذاری دارایی با عوامل بیشتر در بورس تهران (داده تلفیقی) | اقتصاد مقداری 7(4) |
| 121 | بهمنی، محمدپورزرندی & مینویی (1403/2024) | پیشبینی بازدهی سهام در سطح شرکت: پیوند مدلهای قیمتگذاری و عوامل اقتصادی | مهندسی مالی و مدیریت اوراق بهادار 58 |
| 122 | صادقی & کمالی دولتآبادی (1403/2025) | پیشبینی قیمت سهام با رویکرد رگرسیون لاسو در TSE | مهندسی مالی و مدیریت اوراق بهادار 61 |

---

## PART 3. SYNTHESIS — GAP ANALYSIS FOR THE REPLICATION

### What exists on TSE
1. **Parametric SDF estimation exists** (Taleblou cluster: behavioral SDF, sentiment SDF, beta-vs-SDF efficiency on FF 25 portfolios; Davallou & Badri idiosyncratic risk) — but **all GMM on hand-specified kernels. No neural-network SDF.**
2. **Deep learning return prediction is mature** (LSTM/CNN/GRU/TFT/autoencoders, index and firm level, 2003–2026) — DL infrastructure on TSE data is established.
3. **Factor models are well tested** (FF3/FF5/Carhart with GRS, anomaly discovery, LASSO) — test-asset and benchmark infrastructure exists.

### The gap (our paper's contribution)
- **No deep-learning SDF / deep factor model paper exists for TSE (or any frontier/MENA market).** No Iranian paper estimates an SDF from neural-network conditional moment conditions; no autoencoder/latent-factor SDF (CPZ/GKX style); no DL cross-sectional asset-pricing evaluation using GRS or HJ-distance on TSE test portfolios.
- No paper builds anomaly characteristic portfolios at scale for TSE (Naderi Beni et al. is firm-level Bayesian, not long-short portfolios).
- International applications to EMs exist (China: Leippold et al.; broad EM: Hanauer & Kalsbach) but **none for a sanctions-affected frontier market with 5% price bands** — the institutional stress-test angle is unique.

### Core citation set for the paper's lit review (12)
Gu-Kelly-Xiu 2020 · CPZ 2024 · Gu-Kelly-Xiu 2021 · Lettau-Pelger 2020 · Hansen-Jagannathan 1991 · Cochrane 2011 · Kelly-Pruitt-Su 2019 · Kozak-Nagel-Santosh 2020 · Feng-Giglio-Xiu 2020 · Nagel 2021 · Leippold-Wang-Zhou 2022 · Hanauer-Kalsbach 2023. Secondary: Tobek-Hronec 2021, Chen-Zimmermann 2022, Kelly-Xiu 2023, Giglio-Kelly-Xiu 2022, Taleblou et al. (Iranian SDF cluster).

---

## APPENDIX — Working notes

- **Full metadata + NoorMags IDs** for Persian papers: `notes/noormags_findings.md`
- **SID IDs** for Persian papers available on request (agent-verified); PDF downloads require credit on NoorMags/SID.
- **Year convention:** Persian journals publish in Shamsi years; convert when citing (1403 SH ≈ 2024–25).
- **TSE-specific data issues** affecting any replication: 5% daily price bands, thin trading, Persian fiscal year (21-Mar) formation alignment, sanctions-driven regime breaks — see PLAN.md §4.
