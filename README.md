# 🏠 Real Estate Price Prediction - Machine Learning

> Aplicație inteligentă pentru estimarea prețului locuințelor din București folosind Machine Learning și Deep Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-yellow.svg)](https://scikit-learn.org/)

## 📋 Cuprins

- [Despre Proiect](#-despre-proiect)
- [Caracteristici](#-caracteristici)
- [Tehnologii Utilizate](#-tehnologii-utilizate)
- [Structura Proiectului](#-structura-proiectului)
- [Instalare](#-instalare)
- [Utilizare](#-utilizare)
- [Modele și Evaluare](#-modele-și-evaluare)
- [Dataset](#-dataset)
- [Interfața Web](#-interfața-web)
- [Contribuții](#-contribuții)
- [Licență](#-licență)

## 🎯 Despre Proiect

Această aplicație folosește algoritmi de Machine Learning pentru a estima prețul locuințelor din București pe baza mai multor caracteristici:
- Număr de camere
- Suprafață utilă și construită
- An de construcție
- Număr de băi
- Etaj și număr de etaje
- Număr de garaje
- Zona/cartierul

Proiectul include:
- ✅ **Preprocesare avansată** a datelor
- ✅ **Analiză exploratorie** cu vizualizări
- ✅ **3 modele ML/DL**: Regresie Liniară, Arbore de Decizie, Rețea Neuronală
- ✅ **Interfață web interactivă** cu Streamlit
- ✅ **Predicții batch** pentru multiple proprietăți simultan
- ✅ **Explorare date** în timp real

## ✨ Caracteristici

### 🔧 Preprocesare Date
- Curățare și tratarea valorilor lipsă
- Transformare logaritmică pentru preț (reduce influența outlier-ilor)
- Encoding pentru variabile categoriale (zone/cartiere)
- Scalare cu StandardScaler pentru features numerice

### 📊 Analiză Exploratorie
- Vizualizare preț mediu în funcție de suprafață
- Top cartiere după prețul pe metru pătrat
- Grafice interactive cu Matplotlib

### 🤖 Modele Machine Learning
1. **Regresie Liniară** - Model principal folosit în aplicație
2. **Arbore de Decizie** - Alternative cu max_depth=6
3. **Rețea Neuronală** - Sequential model cu TensorFlow/Keras

### 🌐 Interfață Web
- **Estimator individual** cu formular interactiv
- **Predicții batch** prin upload CSV
- **Explorare date** cu statistici și grafice
- Design modern cu Streamlit și CSS personalizat

## 🛠️ Tehnologii Utilizate

| Categorie | Tehnologii |
|-----------|-----------|
| **Limbaj** | Python 3.11 |
| **ML/DL** | scikit-learn, TensorFlow/Keras |
| **Procesare Date** | pandas, numpy |
| **Vizualizare** | matplotlib, plotly |
| **Interfață Web** | Streamlit |
| **Persistență** | joblib |

## 📁 Structura Proiectului

```
RealEstate-Price-Prediction-MachineLearning/
│
├── data/                           # Directorul pentru date
│   ├── house_offers.csv           # Dataset brut (București, Sept 2020)
│   ├── bucuresti_ready.csv        # Dataset preprocesat
│   └── scaler.pkl                 # StandardScaler salvat
│
├── models/                         # Modele antrenate
│   ├── model_linreg.pkl           # Model de Regresie Liniară
│   ├── feature_columns.pkl        # Lista de coloane/features
│   └── scaler.pkl                 # Copia scaler-ului pentru interfață
│
├── README+REQ/                     # Documentație și dependențe
│   ├── README.md                  # README original (în română)
│   └── requirements.txt           # Dependențe Python
│
├── 1.preprocesare.py              # Script preprocesare date
├── 2.analiza_exploratorie.py     # Script analiză și vizualizare
├── 3si4.antrenare&evaluare.py    # Script antrenare și evaluare modele
├── 5.interfata.py                 # Aplicație web Streamlit
│
└── README.md                       # Acest fișier
```

## 🚀 Instalare

### Prerequisite
- Python 3.11 sau superior
- pip (package manager)

### Pași de Instalare

1. **Clonează repository-ul**
```bash
git clone https://github.com/andreilupau/RealEstate-Price-Prediction-MachineLearning.git
cd RealEstate-Price-Prediction-MachineLearning
```

2. **Creează mediu virtual**
```bash
python -m venv venv
```

3. **Activează mediul virtual**

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

4. **Instalează dependențele**
```bash
pip install -r README+REQ/requirements.txt
```

## 💻 Utilizare

### Workflow Complet

#### 1. Preprocesare Date
Procesează dataset-ul brut și pregătește datele pentru antrenare:

```bash
python 1.preprocesare.py
```

**Ce face:**
- Încarcă `data/house_offers.csv`
- Curăță și tratează valorile lipsă
- Aplică transformare logaritmică pentru preț
- Creează variabile dummy pentru zone
- Scalează features numerice
- Salvează `data/bucuresti_ready.csv` și `data/scaler.pkl`

#### 2. Analiză Exploratorie (Opțional)
Vizualizează distribuțiile și relațiile din date:

```bash
python 2.analiza_exploratorie.py
```

**Generează:**
- Grafic: Preț mediu în funcție de suprafață
- Grafic: Top 10 cartiere după €/mp

#### 3. Antrenare și Evaluare Modele
Antrenează și compară cele 3 modele:

```bash
python "3si4.antrenare&evaluare.py"
```

**Output:**
```
Regresie liniară
MAE: 0.XXXX
RMSE: 0.XXXX
R2: 0.XXXX

Arbore de decizie
MAE: 0.XXXX
RMSE: 0.XXXX
R2: 0.XXXX

Rețea neuronală
MAE: 0.XXXX
RMSE: 0.XXXX
R2: 0.XXXX
```

**Salvează:**
- `models/model_linreg.pkl` - Modelul de regresie liniară
- `models/feature_columns.pkl` - Lista de coloane
- `models/scaler.pkl` - Copie a scaler-ului

#### 4. Rulare Interfață Web
Pornește aplicația Streamlit:

```bash
python -m streamlit run 5.interfata.py
```

sau:

```bash
streamlit run 5.interfata.py
```

Aplicația se va deschide automat în browser la `http://localhost:8501`

## 📊 Modele și Evaluare

### Metrici de Evaluare

Fiecare model este evaluat folosind:

- **MAE (Mean Absolute Error)** - Eroarea absolută medie
- **RMSE (Root Mean Squared Error)** - Rădăcina erorii pătratice medii
- **R² Score** - Coeficientul de determinare (0-1, mai mare = mai bun)

### Modele Implementate

#### 1. Regresie Liniară
```python
LinearRegression()
```
- Model simplu și interpretabil
- Performanță bună pentru relații liniare
- **Folosit în interfața web**

#### 2. Arbore de Decizie
```python
DecisionTreeRegressor(max_depth=6, random_state=42)
```
- Capturează relații non-liniare
- Max depth limitat pentru a preveni overfitting

#### 3. Rețea Neuronală
```python
Sequential([
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
```
- 2 hidden layers cu ReLU activation
- Optimizer: Adam
- Loss: MSE (Mean Squared Error)
- 30 epoci de antrenare

### Target Prediction

**Important:** Toate modelele prezic `price_log` (logaritm natural al prețului), apoi convertesc înapoi cu `expm1()`:

```python
price_predicted = np.expm1(model.predict(X))
```

Această transformare îmbunătățește performanța și reduce influența valorilor extreme.

## 📈 Dataset

### Sursa
Dataset cu oferte imobiliare din București (Septembrie 2020)

### Features (Caracteristici)

| Feature | Tip | Descriere |
|---------|-----|-----------|
| `price` | Numeric | Prețul în EUR (target) |
| `location_area` | Categoric | Zona/cartierul |
| `rooms_count` | Numeric | Număr de camere |
| `useful_surface` | Numeric | Suprafață utilă (mp) |
| `built_surface` | Numeric | Suprafață construită (mp) |
| `construction_year` | Numeric | Anul construcției |
| `bathrooms_count` | Numeric | Număr de băi |
| `level` | Numeric | Etajul |
| `max_level` | Numeric | Număr total de etaje |
| `garages_count` | Numeric | Număr de garaje |

### Preprocesare Aplicată

1. **Tratarea valorilor lipsă:**
   - Location: „Unknown"
   - Numeric: mediana coloanei

2. **Feature Engineering:**
   - `price_log = log1p(price)` - transformare logaritmică
   - One-hot encoding pentru `location_area`

3. **Scalare:**
   - StandardScaler pentru toate features numerice
   - Target (`price_log`) nu este scalat

## 🎨 Interfața Web

### Pagini Disponibile

#### 1. 📍 Estimator
- Formular interactiv cu toate caracteristicile
- Selectare zonă din listă
- Rezultat instant cu preț estimat
- Calculare preț per metru pătrat

#### 2. 📊 Predicții Batch
- Upload CSV cu multiple proprietăți
- Descărcare template exemplu
- Procesare automată
- Download rezultate în CSV

#### 3. 🔍 Explorare Date
- Statistici generale (număr anunțuri, preț median, €/mp median)
- Top N cartiere după €/mp
- Grafice interactive cu Plotly
- Preview date brute

#### 4. ℹ️ Despre
- Informații despre model
- Tips și sfaturi
- Documentație rapidă

### Design

- **UI Modern** cu CSS personalizat
- **Responsive layout** cu Streamlit columns
- **Gradient backgrounds** și efecte vizuale
- **Interactive charts** cu Plotly Express

## 🔄 Workflow Complet - Exemplu

```bash
# 1. Activează mediul virtual
venv\Scripts\activate  # Windows
# sau
source venv/bin/activate  # Linux/Mac

# 2. Preprocesează datele
python 1.preprocesare.py

# 3. (Opțional) Analiză exploratorie
python 2.analiza_exploratorie.py

# 4. Antrenează modelele
python "3si4.antrenare&evaluare.py"

# 5. Pornește interfața web
streamlit run 5.interfata.py
```

## 🤝 Contribuții

Contribuțiile sunt binevenite! Pentru a contribui:

1. Fork repository-ul
2. Creează un branch pentru feature (`git checkout -b feature/AmazingFeature`)
3. Commit modificările (`git commit -m 'Add some AmazingFeature'`)
4. Push pe branch (`git push origin feature/AmazingFeature`)
5. Deschide un Pull Request

### Idei de Îmbunătățiri

- [ ] Adăugare modele avansate (XGBoost, Random Forest, LightGBM)
- [ ] Hyperparameter tuning cu GridSearchCV
- [ ] Validare încrucișată (k-fold cross-validation)
- [ ] Feature importance analysis
- [ ] API REST pentru predicții
- [ ] Docker containerization
- [ ] Deploy pe cloud (Heroku, AWS, Azure)
- [ ] Actualizare dataset cu date recente
- [ ] Predicții pentru alte orașe din România

## 📝 Licență

Acest proiect este dezvoltat în scop educațional.

## 👤 Autor

**Andrei Lupau**

- GitHub: [@andreilupau](https://github.com/andreilupau)

## 📞 Contact

Pentru întrebări sau sugestii, deschide un issue pe GitHub.

---

⭐ Dacă îți place acest proiect, dă-i un star pe GitHub!

**Made with ❤️ and Python**
