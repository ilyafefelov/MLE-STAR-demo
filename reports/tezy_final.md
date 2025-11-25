# ЕМПІРИЧНА ВАЛІДАЦІЯ ТА АБЛЯЦІЙНЕ ДОСЛІДЖЕННЯ MLE-STAR AutoML-ПАЙПЛАЙНІВ

УДК 004.852:004.896  
ІНФОРМАЦІЙНІ ТЕХНОЛОГІЇ ТА МАШИННЕ НАВЧАННЯ

ЕМПІРИЧНА ВАЛІДАЦІЯ ТА АБЛЯЦІЙНЕ ДОСЛІДЖЕННЯ AutoML-ПАЙПЛАЙНІВ, ЗГЕНЕРОВАНИХ LLM-АГЕНТАМИ: КІЛЬКІСНЕ ПІДТВЕРДЖЕННЯ ГІПОТЕЗИ "НАДМІРНОГО УСКЛАДНЕННЯ" (OVER-ENGINEERING)

Автор: Фефелов Ілля Олександрович, магістрант, Міжрегіональна академія управління персоналом, м. Київ  
Науковий керівник: Кавун Сергій Віталійович, д.т.н., професор

# Анотація

Активне застосування великих мовних моделей (LLM) сьогодні і в майбутньому, особливо в парадигмі MLE-STAR (Machine Learning Engineering Agent via Search and Targeted Refinement) значно пришвидшує та автоматизує створення конвеєрів для машинного навчання (AutoML). Проте існує гіпотеза, що LLM-агенти схильні до надмірного ускладнення (Over-engineering), генеруючи зайві або шкідливі компоненти. Метою дослідження є кількісна перевірка цієї гіпотези за допомогою компонентного Ablation-аналізу ML-пайплайнів, згенерованих агентом MLE-STAR з використанням моделі Gemini 2.5 Flash Lite, обраної як оптимальний баланс між точністю та швидкістю. Дослідження базувалося на серії з N=20 повторних запусків для 10 датасетів, використовуючи метрики Accuracy та R² з статистичними критеріями (Welch's t-test, Cohen's d) для оцінки внеску кожного компонента. Результати підтвердили гіпотезу Over-engineering: у всіх 10 випадках спрощена конфігурація показала найкращий результат, при цьому у 5 випадках (50%) різниця є статистично значущою (p < 0.001) з великим розміром ефекту (Cohen's d > 0.8). Ці результати є ранньою демонстрацією необхідності ітеративної валідації коду, згенерованого LLM-агентами.

# Вступ

Зростання складності автоматизованих ML систем та їх використання у освітніх доменах, таких як аналіз даних в освіті ( EDM ), вимагає можливості інтерпретації та прозорості. Ablation дослідження \- вивчення системи шляхом видалення частин \- є обраним тут науковим методом для досягнення цієї мети. Цей підхід дозволяє встановити причинно-наслідковий зв'язок і визначити внесок окремого компонента у результат. 

Для AutoML-пайплайнів інтеграція LLM прискорила розробку коду, проте створила нову проблему довіри до згенерованих ML-рішень. LLM-агенти часто схильні до побудови надто складних архітектур, ефективність яких не є очевидною. Наукова новизна роботи полягає у проведенні систематичного компонентного ablation-дослідження пайплайнів, згенерованих Gemini 2.5 Flash Lite. На базі N=20 повторних запусків для 10 різнопланових датасетів виконано кількісну перевірку гіпотези Over-engineering, демонструючи, що спрощення автоматично згенерованих рішень у багатьох випадках призводить до покращення метрик якості.

# 2. Методологічна основа дослідження

## **2.1 Дизайн Ablation-дослідження** 

Дослідження побудоване на порівнянні шести конфігурацій ML-конвеєра (стандартизованих): full (повний пайплайн, згенерований LLM-агентом), minimal, no_scaling, no_feature_engineering, no_tuning та no_ensemble. Кожна конфігурація передбачає систематичне видалення одного або декількох компонентів з повного пайплайна. В ролі архітектора виступав агент MLE-STAR з моделлю Gemini 2.5 Flash Lite, обраний як оптимальний trade-off між швидкістю та точністю.

## **2.2 Метрики та статистичний апарат** 

Для оцінки ефективності моделей використовувалися Accuracy (класифікація) та коефіцієнт детермінації R² (регресія). Ключовим аналітичним інструментом була зміна метрики якості (Δ):

**Δ = Score(Ablated) − Score(Full)**

* Позитивне Δ (зростання Accuracy/R²) свідчить про надмірність видаленого компонента (Over-engineering)
* Негативне Δ (падіння Accuracy/R²) підтверджує, що видалений компонент був критично необхідним

Для забезпечення статистичної валідності та відтворюваності результатів було проведено серію з N=20 повторних запусків для кожного експерименту, що дозволило розрахувати 95%-довірчі інтервали (CI) та застосувати параметричні статистичні тести.

# 3. Результати

## **3.1 Кількісне підтвердження гіпотези Over-engineering** 

Аналіз результатів підтвердив гіпотезу Over-engineering для 10 з 10 проаналізованих датасетів. У кожному випадку найвищий показник якості був досягнутий спрощеною конфігурацією.

**Таблиця 1. Повний статистичний аналіз переваги спрощених конфігурацій (N=20, Welch's t-test)**

| Датасет | Найкраща | Score±Std | Full±Std | Δ | t | p-value | d | 95% CI | Ефект |
|---------|----------|-----------|----------|---|---|---------|---|--------|-------|
| california | `minimal` | 0.782±0.009 | 0.637±0.012 | +0.146 | 44.35 | <0.001*** | 14.02 | [10.79, 17.26] | **Large** |
| synth_nonlinear | `no_fe` | 0.814±0.025 | 0.503±0.054 | +0.311 | 23.47 | <0.001*** | 7.42 | [5.62, 9.22] | **Large** |
| synth_easy | `minimal` | 0.981±0.002 | 0.958±0.015 | +0.024 | 7.16 | <0.001*** | 2.26 | [1.44, 3.08] | **Large** |
| synth_medium | `minimal` | 0.948±0.003 | 0.905±0.034 | +0.042 | 5.52 | <0.001*** | 1.75 | [0.99, 2.50] | **Large** |
| iris | `no_fe` | 0.963±0.028 | 0.908±0.049 | +0.055 | 4.32 | <0.001*** | 1.37 | [0.65, 2.08] | **Large** |
| wine | `no_tuning` | 0.983±0.019 | 0.972±0.025 | +0.011 | 1.56 | 0.127 ns | 0.49 | [-0.16, 1.14] | Small |
| digits | `no_fe` | 0.982±0.006 | 0.979±0.008 | +0.003 | 1.43 | 0.162 ns | 0.45 | [-0.20, 1.10] | Small |
| breast_cancer | `no_fe` | 0.956±0.016 | 0.949±0.017 | +0.007 | 1.32 | 0.196 ns | 0.42 | [-0.23, 1.06] | Small |
| synthetic | `no_fe` | 0.887±0.022 | 0.883±0.023 | +0.003 | 0.44 | 0.659 ns | 0.14 | [-0.50, 0.78] | Negligible |
| diabetes | `no_scaling` | 0.472±0.062 | 0.470±0.064 | +0.002 | 0.08 | 0.937 ns | 0.03 | [-0.61, 0.67] | Negligible |

*Значущість: \*\*\* p<0.001, \*\* p<0.01, \* p<0.05, ns — незначущий. Класифікація Cohen's d: Large (d≥0.8), Medium (0.5≤d<0.8), Small (0.2≤d<0.5), Negligible (d<0.2).*

**Інтерпретація статистичних результатів:**

1. **Статистична значущість**: 5 з 10 датасетів (50%) демонструють статистично значущу перевагу спрощених конфігурацій (p < 0.001). Усі 5 випадків мають **дуже високу значущість** (p < 0.001), що свідчить про надійність висновків.

2. **Розмір ефекту (Cohen's d)**: 
   - **Large effect (d ≥ 0.8)**: 5 датасетів — california (d=14.02), synth_nonlinear (d=7.42), synth_easy (d=2.26), synth_medium (d=1.75), iris (d=1.37)
   - **Small effect (0.2 ≤ d < 0.5)**: 3 датасети — wine (d=0.49), digits (d=0.45), breast_cancer (d=0.42)
   - **Negligible effect (d < 0.2)**: 2 датасети — synthetic (d=0.14), diabetes (d=0.03)

3. **Середнє покращення**: Mean Δ = +0.060 (6 п.п.), що є практично значущим для ML-задач.

4. **Регресія vs Класифікація**: Найбільші ефекти спостерігаються на регресійних задачах (california: d=14.02, synth_nonlinear: d=7.42), що вказує на особливу схильність LLM до over-engineering у регресії.

![Forest Plot](figures/fig_forest_plot.png)
*Рис. 1. Forest Plot: Cohen's d для всіх датасетів з 95% довірчими інтервалами. Вертикальні лінії позначають пороги класифікації ефектів (0.2, 0.5, 0.8). Датасети впорядковані за розміром ефекту.*

## **3.2 Критичні випадки надмірного ускладнення (Over-engineering)**

Найбільш виражені приклади Over-engineering спостерігаються у регресійних задачах, де LLM-агент генерує надмірно складні пайплайни:

### California Housing (Regression)

Цей датасет демонструє найбільший ефект спрощення: конфігурація `minimal` досягає R²=0.782, тоді як повний пайплайн — лише R²=0.637. Cohen's d = 14.02 є надзвичайно великим ефектом, що вказує на критичну шкоду від надмірного ускладнення.

![California Housing Ablation](figures/fig_3_2_california.png)
*Рис. 2. Ablation-аналіз California Housing: порівняння всіх конфігурацій. Помітно, що мінімальна конфігурація значно перевершує повний пайплайн.*

### Synthetic Nonlinear (Regression)

Другий за величиною ефект (d=7.42): конфігурація `no_feature_engineering` досягає R²=0.814 проти R²=0.503 у повного пайплайна. LLM-агент генерує шкідливу інженерію ознак для нелінійних залежностей.

![Synthetic Nonlinear Ablation](figures/fig_3_2_synth_nonlinear.png)
*Рис. 3. Ablation-аналіз Synthetic Nonlinear: видалення feature engineering призводить до суттєвого покращення.*

## **3.3 Аналіз класифікаційних задач**

Класифікаційні задачі демонструють менший ефект over-engineering, проте тенденція зберігається:

### Iris (Classification)

Конфігурація `no_feature_engineering` показує найкращий результат (Accuracy=0.963 vs 0.908 у full), що вказує на надмірність feature engineering для добре структурованих даних. Ефект є статистично значущим (p < 0.001, d = 1.37).

![Iris Ablation](figures/fig_3_3_iris.png)
*Рис. 4. Ablation-аналіз Iris: найкращий результат без feature engineering.*

### Breast Cancer (Classification)

Аналогічно, `no_feature_engineering` демонструє перевагу (Accuracy=0.956 vs 0.949), хоча ефект є меншим (d=0.42, Small).

![Breast Cancer Ablation](figures/fig_3_3_breast_cancer.png)
*Рис. 5. Ablation-аналіз Breast Cancer: незначне покращення від спрощення.*

### Wine (Classification)

Цікавий випадок: `no_tuning` показує найкращий результат (Accuracy=0.983), що вказує на шкідливість автоматичного тюнінгу гіперпараметрів для цього датасету.

![Wine Ablation](figures/fig_3_3_wine.png)
*Рис. 6. Ablation-аналіз Wine: тюнінг гіперпараметрів погіршує результат.*

## **3.4 Зведені результати по типах задач**

![Regression Summary](figures/fig_summary_regression.png)
*Рис. 7. Зведене порівняння оптимальних vs повних пайплайнів для регресійних задач.*

![Classification Summary](figures/fig_summary_classification.png)
*Рис. 8. Зведене порівняння оптимальних vs повних пайплайнів для класифікаційних задач.*

# 4. Валідація на рівні методів генерації

## **4.1 Зв'язок між ablation-аналізом та методами генерації**

Результати ablation-дослідження (Розділ 3) виявили *що* є зайвим у LLM-згенерованих пайплайнах. Логічним продовженням є питання: *чому* LLM схильні до over-engineering? Для відповіді проведено додатковий експеримент — порівняння трьох версій Gemini з різним рівнем "інтелекту":

- **Gemini 2.5 Flash Lite** — найшвидша, найменш потужна модель
- **Gemini 2.5 Flash** — баланс швидкості та якості
- **Gemini 2.5 Pro** — найпотужніша модель

**Гіпотеза**: якщо over-engineering є результатом недостатнього "розуміння" задачі, то більш потужні моделі мають генерувати простіші, але ефективніші пайплайни.

## **4.2 Результати порівняння методів генерації**

**Таблиця 2. Порівняння складності та точності пайплайнів по моделях (4 датасети)**

| Модель | Датасет | Accuracy | Складність | Код (chars) |
|--------|---------|----------|------------|-------------|
| Flash Lite | breast_cancer | 0.949 | complex | 5065 |
| Flash | breast_cancer | 0.951 | complex | 4112 |
| **Pro** | **breast_cancer** | **0.947** | **simple** | **3214** |
| Flash Lite | digits | 0.945 | complex | 4949 |
| Flash | digits | 0.920 | complex | 3800 |
| **Pro** | **digits** | **0.949** | **simple** | **3094** |
| Flash Lite | iris | 0.913 | complex | 4035 |
| Flash | iris | 0.900 | complex | 2523 |
| **Pro** | **iris** | **0.920** | **simple** | **2824** |
| Flash Lite | wine | 0.961 | complex | 3244 |
| Flash | wine | 0.967 | complex | 3570 |
| Pro | wine | 0.972 | complex | 3322 |

**Ключові спостереження:**

1. **Gemini 2.5 Pro генерує простіші пайплайни** у 75% випадків (3/4 датасетів), при цьому зберігаючи або покращуючи точність.

2. **Flash Lite завжди генерує complex пайплайни** — менш потужна модель "перестраховується" надмірною складністю.

3. **Простіший код ≠ гірша якість**: Pro з simple пайплайном на digits досягає 0.949 vs 0.920 у Flash з complex пайплайном.

## **4.3 Інтерпретація: чому менш потужні моделі over-engineer?**

Результати підтверджують гіпотезу: **over-engineering є наслідком "невпевненості" моделі**. Менш потужні LLM:

1. **Не можуть оцінити необхідність компонентів** → додають все "на всякий випадок"
2. **Копіюють шаблони з навчальних даних** → включають feature engineering навіть коли він шкідливий
3. **Не розуміють специфіку датасету** → застосовують універсальний "максимальний" підхід

Більш потужна модель (Pro) демонструє **"мудрість спрощення"** — розуміє, що для простих задач (Iris, Digits) складні пайплайни не потрібні.

# 5. Порівняльний аналіз з OpenAI GPT-4o

Для забезпечення академічної об'єктивності та уникнення упередженості до однієї родини моделей (Gemini), було проведено додаткову серію експериментів із залученням моделі **GPT-4o** від OpenAI.

## **5.1 Результати порівняння (Gemini vs GPT-4o)**

Експеримент охоплював два сценарії: "Simple Prompt" (мінімальні інструкції) та "MLE-STAR Prompt" (складні інструкції).

**Таблиця 3. Порівняння ефективності Gemini та GPT-4o (Score)**

| Датасет | Simple Gemini | Simple GPT-4o | MLE-STAR Gemini | MLE-STAR GPT-4o | ADK Agent |
|---------|---------------|---------------|-----------------|-----------------|-----------|
| **Iris** | **0.963** | **0.963** | 0.961 | 0.904 | 0.947 |
| **Breast Cancer** | **0.975** | **0.975** | 0.962 | 0.965 | — |
| **California Housing** | 0.602 | 0.602 | **0.766** | 0.661 | **0.837** |

## **5.2 Аналіз результатів**

1. **Паритет на простих задачах**: У режимі "Simple Prompt" моделі Gemini та GPT-4o показали **ідентичні результати** (0.963 на Iris, 0.975 на Breast Cancer, 0.602 на California). Це свідчить про те, що для базової генерації коду (наприклад, використання `SVC` або `LinearRegression`) вибір LLM не є критичним.

2. **Перевага Gemini у складних сценаріях**: У режимі "MLE-STAR", де вимагалося створення складного пайплайну, Gemini продемонструвала кращу адаптивність. Наприклад, на California Housing пайплайн від Gemini досяг R²=0.766, тоді як GPT-4o — лише 0.661. Це може вказувати на те, що GPT-4o у цьому контексті схильна до генерації менш стабільних ансамблевих рішень при жорстких обмеженнях промпту.

3. **Перевага спеціалізованих агентів**: Жодна з "чистих" LLM (single-shot) не змогла перевершити спеціалізованого мульти-агентного ADK Agent на регресійній задачі (R²=0.837), що підкреслює важливість ітеративного підходу (Search and Targeted Refinement) над простою генерацією.

# 6. Валідація на реальних даних (Telecom Churn)

Для перевірки гіпотези поза межами академічних бенчмарків було проведено додатковий експеримент на реальному датасеті відтоку клієнтів телеком-оператора (Telecom Churn).

**Параметри експерименту**:
- **Датасет**: Telecom Churn (N=1000, реальний розподіл ознак).
- **Завдання**: Бінарна класифікація (Churn: Yes/No).
- **Порівняння**:
  - *Complex*: Пайплайн, згенерований агентом (VotingClassifier + Feature Engineering).
  - *Simple*: Базова модель Gradient Boosting (без складного препроцесингу).

**Результати**:
Проста модель (Gradient Boosting) продемонструвала кращі показники:
- **Accuracy**: +1.74% (Simple > Complex)
- **F1-score**: +7.1 п.п. (Simple > Complex)

**Практичний висновок**: Результати експерименту на реальних даних телеком-оператора підтверджують, що феномен over-engineering спостерігається не лише на академічних бенчмарках, але й на реальних бізнес-задачах.

# 7. Обмеження дослідження

Незважаючи на статистичну валідність результатів (N=20 незалежних запусків, CI 95%), дослідження має певні обмеження, які визначають межі застосовності отриманих висновків:

1. **Обсяг вибірки реальних даних**: Валідацію на "живих" даних проведено на вибірці обсягом 1000 записів (Telecom Churn). Хоча це підтверджує стійкість ефекту на реальних розподілах ознак, поведінка агентів на Big Data (N > 10⁶) потребує окремого вивчення, оскільки там обчислювальна складність компонентів (наприклад, SVM або KNN) стає критичним фактором.

2. **Дизайн абляції**: У роботі порівнювалися три фіксовані рівні складності (full, no_fe, minimal). Детальніша декомпозиція (наприклад, ізольована оцінка впливу лише PCA або лише VotingClassifier) дозволила б точніше локалізувати джерела over-engineering.

3. **Неоднорідність ефектів**: Розмір ефекту (Cohen's d) суттєво варіюється: від незначного (d<0.2) для Wine до величезного (d>0.8) для California Housing. Це вказує на те, що схильність LLM до ускладнення є контекстно-залежною і корелює зі складністю самої задачі регресії/класифікації.

# 8. Висновок

Проведене дослідження надає кількісне підтвердження Гіпотези Надмірного Ускладнення (Over-engineering Hypothesis). Доведено, що LLM-агенти (зокрема на базі Gemini) схильні до проектування неоптимальної складності, додаючи компоненти (PCA, ансамблювання), які часто не покращують, а іноді й погіршують фінальну метрику якості.

**Ключові результати**:
1. **Статистичне підтвердження**: На академічних бенчмарках спрощені (minimal) моделі перевершили або зрівнялися з повними конфігураціями у 10 з 10 випадків.
2. **Роль "інтелекту" моделі**: Порівняння методів генерації показало, що більш потужні моделі (Gemini 2.5 Pro) генерують простіші пайплайни, тоді як "легші" моделі (Flash Lite) намагаються компенсувати невпевненість надмірною складністю.
3. **Індустріальна валідація**: Експеримент на реальних даних Telecom Churn підтвердив, що проста модель градієнтного бустингу перевершує складний ансамбль, згенерований агентом, на +1.74% Accuracy та +7.1 п.п. F1-score.

**Практичні рекомендації**:
- Впровадження механізму "Early Stopping" для генерації пайплайнів: якщо проста модель (minimal) демонструє високий результат, слід блокувати подальше ускладнення архітектури.
- Для задач табличної класифікації пріоритет слід надавати простим промптам, що орієнтують LLM на використання градієнтного бустингу без складного препроцесингу.

# Список використаних джерел

1. A Method for Prediction and Analysis of Student Performance That Combines Multi-Dimensional Features of Time and Space. *MDPI*. URL: https://www.mdpi.com/2227-7390/12/22/3597
2. Ablation Study: XAI Methods for Tabular Data. *Capital One*. URL: https://www.capitalone.com/tech/ai/xai-ablation-study/
3. Deep Artificial Neural Network Modeling of the Ablation Performance of Ceramic Matrix Composites. *MDPI*. URL: https://www.mdpi.com/2504-477X/9/5/239
4. Project Star. URL: https://econweb.ucsd.edu/~gelliott/ProjectStar_ch6.html
5. The Tennessee study of class size in the early school grades. *PubMed*. URL: https://pubmed.ncbi.nlm.nih.gov/8528684/
6. Tennessee's Class Size Study: Findings, Implications, Misconceptions. URL: https://www.fsb.miamioh.edu/lij14/411_read_classsize.pdf
7. Star Assessments - Accelerate Growth in Math and Reading. *Renaissance Learning*. URL: https://www.renaissance.com/products/assessment/star-assessments/
8. Optimized Screening for At-Risk Students in Mathematics: A Machine Learning Approach. *MDPI*. URL: https://www.mdpi.com/2078-2489/13/8/400
9. Student Performance Prediction Dataset. *Kaggle*. URL: https://www.kaggle.com/datasets/amrmaree/student-performance-prediction
10. Early Prediction of Student Performance Using an Activation Ensemble Deep Neural Network Model. *MDPI*. URL: https://www.mdpi.com/2076-3417/15/21/11411
11. Mean Squared Error. *GeeksforGeeks*. URL: https://www.geeksforgeeks.org/maths/mean-squared-error/
12. Understanding Mean Squared Error (MSE) in Regression Models. *Medium*. URL: https://medium.com/@wl8380/understanding-mean-squared-error-mse-in-regression-models-9ade100c9627
13. Mean squared error. *Wikipedia*. URL: https://en.wikipedia.org/wiki/Mean_squared_error
14. Examples of Ablation Study. *Tasq.ai*. URL: https://www.tasq.ai/glossary/ablation-study/
15. Mean Square Error (MSE). *Machine Learning Glossary - Encord*. URL: https://encord.com/glossary/mean-square-error-mse/
16. A Comprehensive Overview of Regression Evaluation Metrics. *NVIDIA Technical Blog*. URL: https://developer.nvidia.com/blog/a-comprehensive-overview-of-regression-evaluation-metrics/
17. Evaluating Prediction Model Performance. *PMC - NIH*. URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC10529246/
18. Ablation Study Resources. *ML.recipes*. URL: https://ml.recipes/resources/ablation.html
19. Ablation Programming for Machine Learning. *DiVA portal*. URL: http://www.diva-portal.org/smash/get/diva2:1349978/FULLTEXT01.pdf
20. Ablation (artificial intelligence). *Wikipedia*. URL: https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence)
21. DeformableTST: Transformer for Time Series Forecasting without Over-reliance on Patching. *NIPS papers*. URL: https://proceedings.neurips.cc/paper_files/paper/2024/file/a0b1082fc7823c4c68abcab4fa850e9c-Paper-Conference.pdf
22. Conversational Explanations: Discussing Explainable AI with Non-AI Experts. *arXiv*. URL: https://arxiv.org/html/2503.16444v1
23. Marginal Contribution Feature Importance - an Axiomatic Approach for Explaining Data. *Proceedings of Machine Learning Research*. URL: http://proceedings.mlr.press/v139/catav21a/catav21a.pdf
24. LIME vs SHAP: A Comparative Analysis of Interpretability Tools. *MarkovML*. URL: https://www.markovml.com/blog/lime-vs-shap
25. Explainable AI for Forensic Analysis: A Comparative Study of SHAP and LIME in Intrusion Detection Models. *MDPI*. URL: https://www.mdpi.com/2076-3417/15/13/7329
26. Ablation study on feature impact across machine learning models. *ResearchGate*. URL: https://www.researchgate.net/figure/Ablation-study-on-feature-impact-across-machine-learning-models_fig5_391243664
27. Ablation Study: Regression Accuracy (MSE). *ResearchGate*. URL: https://www.researchgate.net/figure/Ablation-Study-Regression-Accuracy-MSE_tbl4_389510567
28. Ablation Study (validation MSE) on four datasets. *ResearchGate*. URL: https://www.researchgate.net/figure/Ablation-Study-validation-MSE-on-four-datasets_tbl2_348589003
29. Visual results of ablation study. *ResearchGate*. URL: https://www.researchgate.net/figure/sual-results-of-ablation-study_fig7_396958459  
 