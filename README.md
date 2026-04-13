# Efficient Duplicate Detection

## 📌 Projektbeschreibung

Dieses Projekt beschäftigt sich mit der **Erkennung von Duplikaten in Produktdaten**.

Ziel ist es, automatisch zu bestimmen, welche Datensätze **dieselbe reale Entität beschreiben**, obwohl sie unterschiedlich formuliert oder strukturiert sind.

Die Daten stammen aus verschiedenen Produktdomänen und enthalten stark variierende und teilweise unstrukturierte Beschreibungen.

---

## 🎯 Ziel

Das Hauptziel des Projekts ist die **Identifikation von Duplikaten innerhalb und zwischen Datensätzen**.

Dabei sollen Einträge erkannt werden, die trotz unterschiedlicher Darstellung:
- dasselbe Produkt repräsentieren
- ähnliche oder gleiche Eigenschaften besitzen
- aus unterschiedlichen Quellen stammen können

---

## 📊 Eingabeformat der Daten

Es werden zwei unterschiedliche Datentypen verwendet:

### 1. Produktbeschreibungen (unstrukturiert)
Die Daten bestehen aus:
- einer eindeutigen ID
- einer freien Textbeschreibung eines Produkts

Diese Beschreibungen enthalten typischerweise:
- technische Spezifikationen
- Produktnamen
- Händler- oder Quelleninformationen
- unstrukturierte und inkonsistente Schreibweisen

---

### 2. Strukturierte Produktdaten
Die Daten bestehen aus mehreren Feldern, typischerweise:

- eindeutige ID
- Produktname
- Preis
- Hersteller / Marke
- optionale Beschreibung
- optionale Kategorie

Ein Teil der Felder kann leer oder unvollständig sein.

---

## 🧠 Herausforderungen

Die Datensätze enthalten typische Probleme aus realen Produktdaten:

- stark variierende Schreibweisen
- Mehrsprachigkeit
- fehlende oder unvollständige Attribute
- inkonsistente Formatierungen
- Rauschen in den Produktbeschreibungen

---

## ⚙️ Methodischer Ansatz

Zur Erkennung von Duplikaten werden typische Verfahren aus dem Bereich **Text Similarity und Record Linkage** eingesetzt, darunter:

- Textvorverarbeitung und Normalisierung
- Vergleich von Attributen und Textinhalten
- Ähnlichkeitsmaße auf Basis von jaccard
- Schwellenwertbasierte Entscheidungslogik

---

## 🚀 Ergebnis

Am Ende entsteht ein System, das in der Lage ist:

> automatisch zu bestimmen, ob zwei Produktdatensätze dasselbe reale Produkt beschreiben

---