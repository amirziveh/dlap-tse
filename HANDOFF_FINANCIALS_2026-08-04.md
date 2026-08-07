# HANDOFF — تکمیل دیتای فایننشال ۳ کشور (IR/TR/PK) — 2026-08-04

**این فایل رو اول بخون.** کار اصلی پروژه (مدل/مقاله) توی `HANDOFF.md` — این فایل فقط وضعیت **دیتای فایننشال بینکشوری** رو داره.

---

## TL;DR

**هدف:** کامل کردن دیتای فایننشال سالانه ۳ کشور (ایران/ترکیه/پاکستان) روی اسکیمای ۱۶ فیلدی:
`country, symbol, year, TA, TL, Eq, CA, CL, Sales, COGS, GP, PAT, Cash, Inv, PPE, dividends`

**وضعیت:** ✅ ایران و ترکیه **تموم شدن** (اسکریپتهای قابل بازتولید + QA). ⏭️ پاکستان = قدم بعدی (پیچیدهتر، ~$1–1.5، ۱–۲ ساعت). ⏭️ بعدش v3 merge + فیلتر مالی + آپلود گوگلشیت.

---

## ۱) خروجیهای نهایی (همین الان روی دیسک)

| کشور | فایل | ردیف | پوشش | اسکریپت |
|---|---|---|---|---|
| 🇮🇷 ایران | `data_ir/financials_annual.csv` | ۹٬۸۸۸ | 16/16 فیلد: BS ~99.9%، Cash 99.8%، PPE 99.8%، Inv 87.9%، Sales 87.2%، COGS 85.0%، GP 94.8%، dividends 63.3% | `scripts/build_ir_financials.py` |
| 🇹🇷 ترکیه | `data_tr/financials_annual.csv` | ۵٬۹۰۷ | 12/12 فیلد **100%**، dividends 85.1% | `scripts/build_tr_financials.py` |
| 🇵🇰 پاکستان | `data_pk/financials_annual.csv` | ۳٬۳۵۸ (قدیمی) | 40–60% — **نیاز به re-extract** | `scripts/pk_vlm_pipeline.py` |

**نسخه v2 قبلی (برای مقایسه):** `/tmp/combined_financials_v2.csv` — ۲۰٬۲۳۳ ردیف (IR 9,888 + PK 4,349 + TR 5,996) — همونیه که توی گوگلشیت آپلود شده (نسخه واسط، باگها توش هستن).

---

## ۲) 🇮🇷 ایران — کامل شد

**منبع:** `fama-five/data/processed/accounting_panel.csv` (پنل انگلیسیشده رهآورد، ۲۸ فیلد) + `fama-five/data/rahavard_unified.csv` (خطوط فارسی ترازنامه) + `fama-five/data/rahavard_dps.csv` (سود هر سهم).

**منطق (مهم برای بازتولید):**
- ۱۲ فیلد پایه از `accounting_panel` — عیناً منطق v2 (تأیید: ۸۵٬۵۳۷ مقدار = ۱۰۰٪ تطابق)
- نگاشت خطوط فارسی: Cash ← «وجوه نقد و موجودیهای نزد بانک» (9,992) · Inv ← «موجودی مواد و کالا» (8,788) · PPE ← «اموال ماشین آلات و تجهیزات» (9,987)
- بکفیل Sales/COGS از «درآمد حاصل از خدمات و فروش» / «بهای تمام شده کالای فروش رفته» (+49/+78 ردیف)
- **dividends = pure_dps × capital / 1e9** (میلیون ریال) — dedup با آخرین `announcement_date`
  - ⚠️ **باگ کشفشده:** تقسیم بر 1e9 درسته نه 1e6 (سهم = سرمایه/1000) — گیت par=1000 (۸۷.۶٪ تطابق؛ بقیه = اثر restatement)
  - سال مالی dps میلادیه → تبدیل شمسی با `jdatetime` (توی venv نصب شد)
- dedup رهآورد: گزارش سالانه (مجمع) + آخرین `report_date` (همون منطق `parse_r365_accounting.py`)

**QA:** Eq=TA−TL فقط ۳ مغایرت (همه ۱۹۹۳، قدیمی) · GP≠Sales−COGS در ۱۲۶ ردیف (۱.۵٪ — از قبل توی v2 بود، همون مقادیر) · spot-check آباد 2015 Cash=13,145 ✓ مطابق منبع خام.

---

## ۳) 🇹🇷 ترکیه — کامل شد + ۴ باگ v2 فیکس شد

**منبع:** `data_tr/financials/<TICKER>/<YEAR>.json` — از İş Yatırım MaliTablo API (کدهای KAP taxonomy).

**کدهای KAP (تأیید شده):** TA=`1BL` · CA=`1A` · CL=`2A` · TL=`2ODB−2N` (کد مستقیم نیست!) · Eq=`2N` · Cash=`1AA` · **Inv=`1AF` (Stoklar)** · PPE=`1BG` · Sales=`3C` · COGS=`3CA` (منفی در KAP → abs) · GP=`3D` · PAT=`3L` · dividends=`4CBB` (منفی در KAP → abs)

**باگهای v2 که فیکس شد:**
1. **Inv از منبع نامعلوم میومد** (مقدارش هیچجا توی JSON نیست، REIT ها موجودی داشتن!) → `1AF` درست
2. **COGS منفی** بود (روش KAP) → abs
3. **dividends منفی** بود → abs
4. ۸۹ ردیف بیمعنی (بیمه/بانک/REIT بدون ترازنامه — AGESA و همکاران) حذف شد (core gate: TA+Eq+Sales+PAT لازم)

**QA:** ۱۰ فیلد = ۰ diff عددی با v2 · Eq=TA−TL = ۰ خطا · GP mismatch ۱۰۷ (۱.۸٪، هلدینگها — مقادیر همان گزارششده، مستند).

**❌ محدودیت باقیمانده:** dividends ۸۷۵ ردیف (2013–2015) خالیه — فایلهای İş Yatırım اون سالها **بخش جریان نقدی ندارن** (تأیید شد: 4C از 2016 اومده). KAP هم قفله (بخش ۵).

---

## ۴) 📋 KAP جدید — مستندات هک API (برای تلاش آینده)

- بکاند: `kapsitebackend.mkk.com.tr` — **DNS عمومی نداره (NXDOMAIN)** → فقط از `www.kap.org.tr/tr/api/...` (پروکسی Next.js)
- **هدر `RSC: 1` الزامیه** — بدونش 404/405؛ بدون Content-Type درست → HTTP 666 (WAF)
- کوکی `KAP=...` بعد از بازدید صفحه ست میشه
- ✅ **کار میکنه:** `POST /tr/api/search/combined` با `{"keyword":"FROTO","discClass":"ALL","lang":"tr","channel":"WEB"}` → oid شرکت (`memberOrFundOid`)
- ✅ **کار میکنه:** `GET /tr/api/company/items/{group}/{state}` → لیست شرکتها (مثلاً IGS/A) — ۷۶۰ شرکت استخراج شد → `/tmp/kap_company_oids.json`
- ✅ **کار میکنه:** `GET /tr/api/menu/about-pdp` (نمونه)
- ❌ `POST /tr/api/disclosure/members/byCriteria` (مسیر درست — از ماژول 92497 توی chunk `0.71jgxax-7yc.js`) → همیشه 500 حتی با payload عیناً مثل سایت (شاید به token/session داخلی نیاز داره)
- ❌ `POST /tr/api/disclosure/list/main` → 400 (مسیر اشتباه بود)
- API قدیمی KAP (`POST /tr/bildirim-sorgu` با JSON) → HTML جدید برمیگردونه
- صفحه SSR `bildirim-sorgu-sonuc?m={oid}&t=...` دیتای اعلانها رو جاسازی میکنه ولی پارامتر تاریخ پیدا نشد (پیشفرض = اخیر)

**نتیجه:** dividends 2013–2015 ترکیه از منابع خودکار فعلی **غیرقابل بازیابیه**. پذیرفته شد + مستند (طبق پلن).

---

## ۵) ⏭️ قدم بعدی: پاکستان (re-extract ۱٬۴۱۳ شرکت-سال failed)

**چرا پیچیدهتر:** VLM + PDF + چند مرحله (داوری صفحه → استخراج → QA). پلن قبلی کامل: `~/.hermes/plans/20260804_010000-dlap-tse-pk-financials-vlm.md` — اول کامل بخون.

**وضعیت:** `data_pk/vlm_state.json` — ۵٬۰۰۹ شرکت-سال: 3,596 done / 1,413 failed. فایل نهایی فعلی `data_pk/financials_annual.csv` = 3,358 ردیف (بعد از حذف بانک/بیمه/مضاربه + حذف ردیفهای بیحاصل PAT/Sales).

**روش re-extract:**
- همون `scripts/pk_vlm_pipeline.py` — فقط state های failed رو دوباره بزن (یا از اول اجرا کن — resumable)
- **مدلها (مهم — جابهجا نشه):** داوری صفحه = `qwen/qwen3.7-flash` چندتصویری **max_tokens≥2048** (مدل استدلالیه — کم باشه → content=null) · استخراج = `nex-agi/nex-n2-mini` تکتصویری **پرامپت کوتاه** (بلند → null)
- کلید: `OPENROUTER_API_KEY` در ~/.bashrc
- هزینه: ~$0.0008/شرکت-سال → ۱٬۴۱۳ شرکت ≈ **$1–1.5**
- ⚠️ دیسک فقط ~4GB: دانلود PDF → استخراج → حذف
- پورتال PSX ریکوست پایتون رو ریجکت میکنه (TLS) → curl
- PDF تست ۴ شرکت: `/tmp/pk_e2e/{ABOT,NML,OGDC,HUBC}/` · مرجع: `data_pk/p0_e2e_{ABOT,NML,OGDC,HUBC}_raw.json` · گیت: تطابق ≥95%

**بعد از پاکستان → v3:**
1. merge سه کشور → `/tmp/combined_financials_v3.csv`
2. **تصمیم باز:** فیلتر بخش مالی (بانک/بیمه/REIT) روی هر ۳ کشور — الان فقط PK فیلتر شده، IR و TR هنوز دارن. پیشنهاد: اعمال کن (طبق تصمیم قبلی «همه بازارها») — ولی PK نهایی 3,358 هم در نظر بگیر (در شیت v2 نسخه واسط 4,349 بود)
3. آپلود گوگلشیت v3 (کامپوزیو فعال: `dlap-sheets`، پوشه DLAP-Backup)

---

## ۶) نکات و تلهها (درسهای این سشن)

- **جابهجایی واحدها:** TR = TL خام · PK = PKR · IR = میلیون ریال — در merge به نسبت تبدیل نزن، هر کشور همون واحد خودش (کاربر نسبتگیری رو بعداً توی ویژگیها میکنه)
- **jdatetime** توی venv نصب شد (برای dps ایران)
- **سود تقسیمی:** هر کشور تعریف متفاوت داره: IR = اعلامی (DPS×سرمایه) · TR 2016+ = پرداختی (4CBB) · PK = پرداختی (PDF) — در مقاله/مستند ذکر کن
- مغایرت شیت v2: PK توی شیت 4,349 ردیفه ولی نهایی لوکال 3,358 — شیت نسخه قبل از فاینالایز رو داره
- اسکریپتها قابل بازتولیدند: `build_ir_financials.py` و `build_tr_financials.py` از روی فایلهای خام کامل بازسازی میکنن

## ۷) وضعیت پلن
`~/.hermes/plans/20260804_160000-dlap-complete-financials-3c.md` — بخش ۷ وضعیت اجرا بهروز شده.
