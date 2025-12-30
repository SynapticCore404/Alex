# Commentator Bot

Telegram channel postlariga insoniy ohangda qisqa komment yozuvchi bot.

- Channel post discussion guruhga avtomatik forward bo‘lganda, post matn/caption Gemini AI’ga yuboriladi.
- Model 1–2 gapli, mantiqli va tabiiy izoh qaytaradi. Post tili qanday bo‘lsa, javob ham shu tilda bo‘ladi.
- Ixtiyoriy `TARGET_CHANNEL_ID` bilan faqat bitta kanal uchun ishlashni cheklash mumkin.

## Talablar
- Python 3.10+
- Telegram bot token (BotFather)
- Google Gemini API key (ai.google.dev)

## O‘rnatish
```powershell
# (ixtiyoriy) virtual muhit
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# kutubxonalar
pip install -r requirements.txt
```

`.env.example` ni nusxa oling va `.env` ga to‘ldiring:
```
TELEGRAM_BOT_TOKEN=...
GEMINI_API_KEY=...
# faqat ma'lum kanal uchun ishlatish (ixtiyoriy)
TARGET_CHANNEL_ID=-1001234567890
```

## Ishga tushirish
```powershell
py main.py
```

## Sozlamalar
- Botni discussion guruhga qo‘shing (minimal: xabar yuborish ruxsati).
- Kanalga discussion guruh ulang (Channel → Manage → Discussion).
- BotFather → /setprivacy → Disable (guruh xabarlarini olish uchun).

## Eslatma (xavfsizlik)
- `.env` faylini hech qachon version control’ga qo‘shmang.
- Agar API kalitlaringiz tasodifan oshkor bo‘lsa, ularni almashtiring (rotate).
