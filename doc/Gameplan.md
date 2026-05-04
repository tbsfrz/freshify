# Research & Strategie — Food Freshness Classifier

## Datasets & Label Mapping

**Kaggle Obst/Gemüse** https://www.kaggle.com/datasets/ulnnproject/food-freshness-dataset
→ Klasse 1+3 vorhanden, Klasse 2 fehlt komplett
- Gurke (V/TF)
- Apfel (V/TW)
- Banane (TW/V)
- Tomaten (V/TF)
- Kartoffeln (TW/TF)
- Erdbeeren (TW/TF)
- Karotten (TF)
- Orange (V)
- Zitronen (TF/TW)

**Kaggle Meat** https://www.kaggle.com/datasets/vinayakshanawad/meat-freshness-image-dataset
→ alle 3 Klassen vorhanden (fresh / half-fresh / spoiled)

**Mendeley Brot** → Klasse 1+3 vorhanden, Klasse 2 fehlt
- Brötchen

**Käse** → kein Dataset vorhanden, alles selbst fotografieren

> Alle Datasets auf 3 Klassen vereinheitlichen: **Frisch / Grenzwertig / Schlecht**
> Zeitstempel bzgl. Tag beim Labeln festhalten

---

## Background Augmentation

- Weißen Hintergrund aus Kaggle/Mendeley Bildern entfernen (rembg Library)
- Lebensmittel auf zufällige echte Hintergründe kleben
- Schließt Domain Gap teilweise — aber nicht vollständig, echte Handyfotos bleiben Pflicht

---

## Two-Stage Fine-Tuning

- **Stage 1** → großer gemischter Datensatz: Kaggle + Mendeley + Background Aug + eigene Fotos für Lücken → Modell lernt Frische als Konzept
- **Stage 2** → ausschließlich eigene Handyfotos, alle Kategorien, alle 3 Klassen, kleinere Learning Rate → Modell spezialisiert sich auf echten Use Case
- Warum Stage 2 sauber ist: alle Klassen haben denselben echten Hintergrund → kein "Hintergrund = Klasse" Problem

---

## Fotoaufwand — 6 Wochen

**Stage 1 — Lücken füllen:**
- Obst/Gemüse Klasse 2 → 50 Fotos
- Brot Klasse 2 → 50 Fotos
- Käse alle 3 Klassen → 150 Fotos
- Fleisch → 0 Fotos (Dataset vollständig)
- **Stage 1 Total: 250 Fotos**

**Stage 2 — alle Kategorien alle Klassen als Handyfotos:**
- Obst/Gemüse K1+K3 → 60 neue Fotos (K2 aus Stage 1 wiederverwendbar)
- Brot K1+K3 → 60 neue Fotos (K2 aus Stage 1 wiederverwendbar)
- Fleisch alle 3 Klassen → 90 neue Fotos
- Käse → aus Stage 1 wiederverwendbar, 0 neue
- **Stage 2 Total: 210 neue Fotos**

**Grand Total: ~460 eigene Fotos**
3 Leute × 6 Wochen = **~26 Fotos pro Person pro Woche**

> Beim Fotografieren immer variieren: verschiedene Hintergründe, Winkel, Lichtverhältnisse — 20 Fotos von einem Apfel sind ok wenn sie wirklich divers sind, sonst bringen sie kaum Mehrwert.

---

## Lebensmittelverschwendung minimieren

- **Aging Protocol** → 1 Item kaufen, alle 2 Tage fotografieren während es altert → ein Apfel deckt automatisch alle 3 Klassen ab ohne Extra-Kauf
- **Supermarkt Restposten** → reduzierte Ware kurz vor Ablauf kaufen → sofort Klasse 3 Material ohne extra warten
- **Kategorien aufteilen** → jede Person übernimmt 1-2 Kategorien komplett → kein doppelter Einkauf
- **Brot und Käse schlau kaufen** → kleine Mengen, mehrere Sorten gleichzeitig in verschiedenen Zuständen fotografieren → ein Kauf, mehrere Datenpunkte
- **Koordination per Gruppe** → wöchentlich abstimmen wer was kauft damit nichts doppelt gekauft wird

---

## TODO

- [ ] Kaggle Food Freshness, Kaggle Meat, Mendeley Brot runterladen und Struktur verstehen — p.P. 60 passende Fotos raussuchen (subjektiv)
- [ ] Streamlit aufsetzen
- [ ] Fotoaufgaben aufteilen: wer fotografiert welche Kategorien ✅
- [ ] Label Mapping definieren: alle Datasets auf 3 Klassen vereinheitlichen
- [ ] Background Augmentation auf Kaggle/Mendeley Bilder anwenden (rembg)
- [ ] Stage 1 trainieren: großer gemischter Datensatz
- [ ] Stage 2 trainieren: nur eigene Handyfotos, kleinere Learning Rate
- [ ] Evaluation: Confidence + Accuracy + MAE + Confusion Matrix pro Klasse
- [ ] Streamlit App auseinandersetzen
- [ ] Repo Projektstruktur
- [ ] Repo README

---

## Echter Use Case

Unser Modell wird nicht "besser" als der Mensch sein — allein schon weil wir ein Riechorgan haben.

**Aber:**
- Tafel bekommt täglich hunderte gespendeter Items → aktuell jemand der jedes einzeln anfasst und schaut → mit Multi-Upload oder Video-Stream wird das zu einem einzigen Scan-Prozess
- Ergebnis ist dokumentiert und nachvollziehbar → kein "wer hat das freigegeben?"
- Schwellenwert einstellbar → Tafel kann entscheiden ab welcher Klasse automatisch aussortiert wird

→ **Automation → Zeitersparnis → konsistente Schwellenwerte**

---

## Automation

Multi-Upload oder Video-Stream → Fließband-artige Klassifikation → nicht mehr Item für Item anfassen sondern ganzen Tisch/Kiste auf einmal scannen
