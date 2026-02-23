# Pention-System

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25-orange)
![Docker](https://img.shields.io/badge/Docker-Enabled-blueviolet)

**Progetto di Tesi – Università degli Studi del Sannio, Corso di Ingegneria Informatica**

Pention-System è una piattaforma avanzata per la simulazione, l’analisi e la localizzazione di sorgenti di emissione tramite dati di sensori distribuiti. Il progetto integra modelli di machine learning e fisica computazionale per stimare la dispersione di sostanze nell’aria, rendendolo un potente strumento per studi ambientali e di sicurezza industriale.

---

## 🚀 Features Principali

- **Simulazione della dispersione**: utilizzo di modelli Gaussian Puff per la predizione della distribuzione di sostanze nell’aria.
- **Localizzazione della sorgente**: stima della posizione di emissioni tramite dati sensoriali.
- **Classificazione NPS**: implementazione di modelli DNN e Balanced Random Forest per l’analisi di spettri di sostanze.
- **Interfaccia interattiva**: visualizzazione dei dati e delle mappe tramite Streamlit e Folium.
- **Robustezza e scalabilità**: struttura modulare e gestione dei dati tramite file LFS.

---

## 📂 Struttura del progetto

```
Pention-System/
│
├─ ClassificatoreNPS/ # Modelli e script per la classificazione NPS
├─ CorrectionDispersion/ # Moduli per la correzione della dispersione simulata
├─ EmissionSourceLocalization/ # Algoritmi per la localizzazione della sorgente
├─ PentionSystem/ # Applicazione principale e API
├─ gaussianPuff/ # Modello di simulazione della dispersione
├─ docker-compose.yml # Configurazione Docker per deploy
├─ .gitignore
├─ .gitattributes
└─ README.md
```

---

## 🛠 Tecnologie Utilizzate

- **Python 3.10.11**
- **TensorFlow / Keras**
- **Scikit-learn**
- **Streamlit & Folium**
- **Docker & Docker-Compose**
- **Git LFS** per gestire file di grandi dimensioni

---

## 💻 Interfaccia Utente

### 🐳 Avvio con Docker

Il progetto è containerizzato per semplificare l’esecuzione e garantire la riproducibilità.

#### 1. Costruzione e avvio dei container

Dalla root del progetto:

```bash
docker-compose up --build
```

#### 2. Streamlit

Per avviare la dashboard locale:

```bash
streamlit run application.py
```

### 📈 Risultati Attesi

- Mappe di concentrazione dinamiche e interattive
- Predizione accurata della posizione della sorgente di emissione
- Classificazione dei dati NPS dai sensori

<img width="1920" height="1080" alt="pention_system_interfaccia" src="https://github.com/user-attachments/assets/86f1cad0-4f3b-45e7-b846-ba86c3e7ce27" />

---

## 📄 License

Questo progetto è rilasciato sotto **MIT License** – vedi il file [LICENSE](LICENSE) per i dettagli.

---

## 💡 Contatti

**Email:** [mininnoclaudio@gmail.com](mailto:mininnoclaudio@gmail.com)

---

*Questo progetto è stato sviluppato come parte della tesi triennale in Ingegneria Informatica presso l’Università degli Studi del Sannio.*
