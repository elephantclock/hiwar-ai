# Hiwar AI — حوار

**Arabic-first multilingual conversational AI for the UAE and GCC.**

Hiwar AI (حوار, "dialogue") is a voice and chat AI agent platform built for how people in the Gulf actually speak: in Arabic dialects, in English, in Hindi and Urdu, and often in several of these within the same sentence. It handles inbound and outbound customer calls, support chats, enquiries, reservations, complaints and service requests.

> **Status:** early design / pre-alpha. The features below describe the target platform.
>
> 🌐 **Website:** https://elephantclock.github.io/hiwar-ai/ · **Early access:** [request here](https://github.com/elephantclock/hiwar-ai/issues/new?template=early-access.yml)

---

## Why Hiwar AI

Most voice AI platforms treat Arabic as one language and assume each call stays in one language. In the GCC, neither is true:

- Customers speak **Gulf dialects**, not Modern Standard Arabic.
- About 85–90% of UAE residents are expatriates. **English and Hindi/Urdu** are everyday service languages.
- People **code-switch** constantly:
  *"abi a7jez table for 4 ba3d el maghrib"* (I want to book a table for 4 after maghrib prayer).

Hiwar AI is designed **Arabic-first**: Arabic dialects, mixed-language speech and Gulf cultural context are core to the system, not added later.

---

## Languages

### 🇦🇪 Arabic: primary language

| Variety | Support |
|---|---|
| **Gulf / Khaleeji Arabic**: Emirati, Saudi (Najdi & Hejazi), Kuwaiti, Qatari, Bahraini, Omani | Understand and speak (primary) |
| **Modern Standard Arabic (MSA / Fusha)** | Understand and speak (formal replies, government, banking) |
| **Egyptian Arabic** | Understand; speak where configured |
| **Levantine Arabic**: Syrian, Lebanese, Jordanian, Palestinian | Understand; speak where configured |
| **Arabizi** (Arabic in Latin letters and numbers, e.g. `7abibi`, `3ala`, `5alas`) | Understand in chat |
| **Gulf Pidgin Arabic** (simplified Arabic mixed with Hindi/Urdu) | Understand |

### Other launch languages

| Language | Support |
|---|---|
| **English**, including Gulf, Indian, Pakistani, Filipino and British accents | Understand and speak |
| **Hindi / Urdu** (Hindustani), in Devanagari, Nastaliq and Roman script in chat | Understand and speak |

### Code-switching

Hiwar AI detects the language of each phrase, not just once per call, and can reply in the customer's own mix:

- **Arabic ↔ English** (Gulf "Arabish")
- **Hindi/Urdu ↔ English** (Hinglish)
- **Arabic ↔ Hindi/Urdu** (Gulf Pidgin)
- **Arabizi ↔ Arabic script ↔ English** in chat

Each client can set the reply style: mirror the customer's mix, formal MSA, Khaleeji only, or English only.

### Roadmap languages

| Phase | Languages |
|---|---|
| **Phase 2** | Malayalam (incl. Manglish), Tagalog (incl. Taglish), Bengali, Persian/Farsi, Tamil |
| **Phase 3** (depends on industry) | Russian, Mandarin, French, German, Turkish, Pashto, Punjabi, Nepali, Sinhala, Telugu, Maghrebi Arabic ↔ French |

---

## Use cases

- **Inbound calls:** enquiries, reservations, order status, complaints, service requests
- **Outbound calls:** booking confirmations, reminders, follow-ups, feedback surveys, collections
- **Customer support chat:** WhatsApp and web widget
- **Human handoff:** passes the call to an agent with a transcript and summary when needed

**Target industries:** hospitality and restaurants, banking and fintech, telecom, government services, real estate, healthcare, retail and e-commerce.

---

## Architecture

```
 Phone (SIP / UAE & KSA numbers)      Chat (WhatsApp, web widget)
            │                                  │
     ┌──────▼───────┐                   ┌──────▼───────┐
     │ Voice gateway│                   │ Chat gateway │
     └──────┬───────┘                   └──────┬───────┘
            │                                  │
     ┌──────▼──────────────────┐        ┌──────▼──────────────┐
     │ Mixed-language speech-   │        │ Text normaliser:     │
     │ to-text (Ar / En / Hi-Ur)│        │ Arabizi → Arabic,    │
     │ + per-phrase lang tag    │        │ Roman Urdu → Urdu    │
     └──────┬──────────────────┘        └──────┬──────────────┘
            └───────────────┬──────────────────┘
                     ┌──────▼──────────────────────────────┐
                     │ Conversation engine (AI model)       │
                     │ • intents: enquiry, reservation,     │
                     │   complaint, service request         │
                     │ • tools: booking, CRM, ticketing     │
                     │ • reply-language rules per client    │
                     └──────┬──────────────────────────────┘
                     ┌──────▼──────────────┐
                     │ Mixed-language TTS   │  Gulf-accent Arabic,
                     │ (voice only)         │  English, Hindi/Urdu
                     └─────────────────────┘

   Shared: recordings & transcripts → labelling → fine-tuning loop,
           analytics, human handoff, UAE/KSA in-country hosting
```

---

## Compliance & data residency

- Hosted in UAE and Saudi cloud regions
- Designed for the **UAE PDPL**, **Saudi PDPL (SDAIA)** and sector regulators (e.g. CBUAE, SAMA)
- Outbound calling follows **TDRA (UAE)** and **CST (KSA)** rules

---

## Roadmap

1. **Phase 0, evaluation:** build a Gulf mixed-language test set and score speech-to-text and text-to-speech engines on it
2. **Phase 1, chat:** WhatsApp and web agent with booking/CRM connections
3. **Phase 2, inbound voice:** live voice loop with replies in under about 800 ms, barge-in (caller can interrupt), human handoff
4. **Phase 3, outbound voice:** reminders, confirmations, follow-ups
5. **Ongoing:** fine-tune speech-to-text and text-to-speech for Gulf dialects and mixed speech using real call data

---

## License

Hiwar AI is **source-available** under the [PolyForm Noncommercial License 1.0.0](LICENSE).

✅ **Free for noncommercial use:** personal study, research, experiments, hobby projects, education, and charitable or public research organisations.

❌ **Commercial use requires a commercial license.** This includes running Hiwar AI for paying customers, using it in your company's contact centre, or offering it as part of a product or service.

### Commercial licensing

For production deployments, enterprise support, or access to Hiwar AI's fine-tuned Gulf Arabic speech models and voices, [request early access](https://github.com/elephantclock/hiwar-ai/issues/new?template=early-access.yml) and we'll follow up about a commercial license.

### Contributing

Contributions are welcome. All contributors must sign a Contributor License Agreement (CLA) so that contributions can be included in both the noncommercial and commercial editions.

---

## Keywords

Arabic voice AI · Gulf Arabic · Khaleeji · Emirati Arabic · Saudi Arabic · conversational AI · AI voice agent · multilingual chatbot · code-switching · Arabizi · Hindi · Urdu · call center AI · customer service automation · WhatsApp bot · UAE · Dubai · Abu Dhabi · Saudi Arabia · GCC
