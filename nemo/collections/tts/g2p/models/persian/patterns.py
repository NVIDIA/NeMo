import re

ZWNJ = chr(0x200C)
SPCE = chr(0x0020)
ZIIR = chr(0x0650)
ZBAR = chr(0x064E)
PISH = chr(0x064F)
TSHD = chr(0x0651) #  _ ّ_

# SOH = chr(0x0001) # start of heading
# EOT = chr(0x0003) # end of text

L03 = r'[بپتثجچحخدذرزژسشصضطظعغفقکگلمنهةئ]'
L04 = r'[ضصثقفغعهخحجچشسیبلتنمکگۀظطزرذدئوةيژؤإأء؟پ]'

CNST = r'[بپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیbptcjCHxdDrzʒsSćĆTZʔGfqkglmnuEi\u200c]'

PUNC = r'[!،؛.؟\u200c\s]+|^|$'


SVOW = r'[َُِaeoęąó]'
LVOW = r'[اآAΛ]'  # how to add ای and او

RegexPatterns = [
    # ZWNJ
        (rf'{ZWNJ}', '-'),
    
    # آ
        (r'آ', 'A'),
        (r'آورد', 'Avard'), 
        (r'آموز', 'Amuz'),
        (r'آمیز', 'Amiz'),
        (r'آلود', 'Alud'),
        (r'آباد', 'AbΛd'),
        (r'آفرین', 'Afarin'),
        (r'آموخته', 'AmuxtE'),
        (r'آگاه', 'AgΛh'),
        (r'آلات', 'AlΛt'),
        (r'آباد', 'AbΛd'),
        (rf'({CNST})(اوری)({PUNC})', r'\1Λvari\3'),

    

    # ا
        (r'ا', 'Λ'),
        (rf'({PUNC})(ای)({CNST})', r'\1I\3'),
        (rf'({PUNC})(ای)({PUNC})', r'\1I\3'),
        (rf'({CNST})(ای)({PUNC})', r'\1Λye\3'),
        (rf'({CNST})(ایی)({PUNC})', r'\1Λyi\3'),
        (rf'({CNST})(ایش)({PUNC})', r'\1ΛyaS\3'),
        (rf'({CNST})(انه)({PUNC})', r'\1ΛnE\3'),
        (rf'({CNST})(اه)({PUNC})', r'\1Λh\3'),
        (rf'({CNST})(ا)({PUNC})', r'\1Λ\3'),
        (rf'({CNST})(ا)({CNST})', r'\1Λ\3'),
    
        (r'اَ', 'ą'),
        (r'اِ', 'ę'),
        (r'اُ', 'ó'),
        (rf'({CNST})(اً)({PUNC})', r'\1Ą\3'),
        (rf'({CNST})(اٍ)({PUNC})', r'\1Ę\3'),
        (rf'({CNST})(اٌ)({PUNC})', r'\1Ó\3'),
    
        (r'انگیز', 'ąngiz'),
    
    # اً
        (r'اً', 'Ą'),
    # اٍ
        (r'اٍ', 'Ę'),
    # اٌ
        (r'اٌ', 'Ó'),
    # اَ
        (r'اَ', 'ą'),
    # اِ
        (r'اِ', 'ę'),
    # اُ
        (r'اُ', 'ó'),
    # ب
        (r'ب', 'b'),
    # پ
        (r'پ', 'p'),
    # ت
        (r'ت', 't'),
        (rf'({LVOW})({CNST})(ت)({PUNC})', r'\1\2t\4'),
        (rf'({CNST})(تان)({PUNC})', r'\1etΛn\3'),
        (rf'({CNST})(تر)({PUNC})', r'\1tar\3'),
        (rf'({CNST})(تری)({PUNC})', r'\1tari\3'),
        (rf'({CNST})(ترین)({PUNC})', r'\1tarin\3'),
    # ث
        (r'ث', 'c'),
    # ج
        (r'ج', 'j'),
    # چ
        (r'چ', 'C'),
    # ح
        (r'ح', 'H'),
    # خ
        (r'خ', 'x'),
        (rf'({PUNC})(خوا)({CNST})', r'\1xΛ\3'),
    # د
        (r'د', 'd'),
    # ذ
        (r'ذ', 'D'),
    # ر
        (r'ر', 'r'),
    # ز
        (r'ز', 'z'),
        (rf'({CNST})(زدگی)({PUNC})', r'\1zadegi\3'),
        (rf'({CNST})(زده)({PUNC})', r'\1zade\3'),
        (rf'({CNST})(زدن)({PUNC})', r'\1zadan\3'),
        (rf'({CNST})(زدا)({PUNC})', r'\1zodA\3'),
    # ژ
        (r'ژ', 'ʒ'),
    # س
        (r'س', 's'),
        (rf'({CNST})(ستان)({PUNC})', r'\1setΛn\3'),
        (rf'({CNST})(شناس)({PUNC})', r'\1SenΛs\3'),
    # ش
        (r'ش', 'S'),
        (rf'({CNST})(ش)({PUNC})', r'\1aS\3'),
        (rf'({CNST})(شان)({PUNC})', r'\1eSΛn\3'),
        (rf'({CNST})(شون)({PUNC})', r'\1eSun\3'),
        (rf'(پوش)({PUNC})', r'puS\2'),
        (rf'(شدگی)({PUNC})', r'Sodegi\2'),
    # ص
        (r'ص', 'ć'),
    # ض
        (r'ض', 'Ć'),
    # ط
        (r'ط', 'T'),
    # ظ
        (r'ظ', 'Z'),
    # ع
        (r'ع', 'ʔ'),
    # غ
        (r'غ', 'G'),
    # ف
        (r'ف', 'f'),
    # ق
        (r'ق', 'q'),
    # ک
        (r'ک', 'k'),
        (rf'({CNST})(کده)({PUNC})', r'\1kadE\3'),
    
    # گ
        (r'گ', 'g'),
        (rf'({CNST})(گر)({PUNC})', r'\1gar\3'),
        (rf'({CNST})(گرا)({PUNC})', r'\1garΛ\3'),
        (rf'({CNST})(گری)({PUNC})', r'\1gari\3'),
        (rf'({CNST})(گاه)({PUNC})', r'\1gΛh\3'),
        (rf'({CNST})(گذاری)({PUNC})', r'\1goDΛri\3'),
        (rf'({CNST})(گی)({PUNC})', r'\1egi\3'),   # زندگی، خستگی
        (r'گذار', 'goDΛr'),
    # ل
        (r'ل', 'l'),
    # م
        (r'م', 'm'),
        (rf'({CNST})(م)({PUNC})', r'\1am\3'),
        (rf'({CNST})(مان)({PUNC})', r'\1emΛn\3'),
        (rf'({CNST})(مند)({PUNC})', r'\1mand\3'),
        (rf'({CNST})(مندی)({PUNC})', r'\1mandi\3'),
        (rf'({CNST})(مون)({PUNC})', rf'\1{ZIIR}mun\3'),
        (rf'({CNST})(مان)({PUNC})', rf'\1{ZIIR}mΛn\3'),

    
    # ن
        (r'ن', 'n'),
        (rf'({CNST})(ند)({PUNC})', r'\1and\3'),
        (rf'({PUNC})(نمی)', r'\1nemi'),
    # و
        (r'و', 'v'),
        (rf'({LVOW})(و)({SVOW}|{LVOW}|{CNST})', r'\1v\3'),
        (rf'({CNST})(ور)({PUNC})', r'\1var\3'),
        (rf'({CNST})(وند)({PUNC})', r'\1vand\3'),
        (rf'({CNST})(وش)({PUNC})', r'\1vaS\3'),
        (rf'({CNST})(وارش)({PUNC})', r'\1varaS\3'),
    # و - u
        (rf'({CNST})(و)({CNST})', r'\1u\3'), # مغلوب

    # و - O 
    # ؤ
        (r'ؤ', 'V'),
        (r'ؤ', 'V'), # و + hamze
    
    # او
        (r'او', 'U'),
    # ه
        (rf'ه', 'h'),
        (rf'({CNST}|{SVOW}|{LVOW})(ه)({CNST})', r'\1h\3'),
        (rf'({CNST})(ه)({PUNC})', r'\1E\3'),
        (rf'({CNST})(های)({PUNC})', r'\1hΛye\3'),
        (rf'({CNST})(هایی)({PUNC})', r'\1hΛyi\3'),
        (rf'({CNST})(هایم)({PUNC})', r'\1hΛyam\3'),
        (rf'({CNST})(هایت)({PUNC})', r'\1hΛyat\3'),
        (rf'({CNST})(هایش)({PUNC})', r'\1hΛyaS\3'),
        (rf'({CNST})(هایمان)({PUNC})', r'\1hΛyemΛn\3'),
        (rf'({CNST})(هایتان)({PUNC})', r'\1hΛyetΛn\3'),
        (rf'({CNST})(هایشان)({PUNC})', r'\1hΛyeSΛn\3'),
        (rf'({CNST})(هایشان)({PUNC})', r'\1hΛyeSΛn\3'),
        (rf'({CNST}|{SVOW}|{LVOW})(هه)({PUNC})', r'\1hE\3'),
    
    # ۀ
        (r'ۀ', 'Y'),
        (r'هٔ', 'Y'), # this is ه + hamze

    
    # ی
        (r'ی', 'i'),
        (rf'({PUNC})(ی)({SVOW}|{LVOW}|{CNST})', r'\1y\3'), # یاشار
        (rf'({SVOW}|{LVOW})(ی)({PUNC})', r'\1y\3'), # صَیاد, آینده
        (rf'({CNST})(ی)({PUNC})', r'\1i\3'), # بستنی
        (rf'({SVOW}|{LVOW})(ی)({SVOW}|{LVOW})', r'\1y\3'), # آیا, ایام
        (rf'({CNST})({SVOW}|{LVOW})(ی)', r'\1\2y'), # تایوان
        (rf'({CNST})(یی)({PUNC})', r'\1yi\3'), # راهنمایی
        (rf'({CNST})(یی)({CNST})', r'\1yi\3'), # تغییرات
        (rf'({CNST})(یم)({PUNC})', r'\1im\3'),
        (rf'({CNST})(ید)({PUNC})', r'\1id\3'),
        (rf'({CNST})(یست)({PUNC})', r'\1ist\3'),
        (rf'({CNST})(یزم)({PUNC})', r'\1izm\3'),
        (rf'({SVOW})(ی)', r'\1y'), # اِیول
        (rf'([آA])(ی)', r'\1y'), # آینده
        (rf'([و])(ی)', r'\1y'), # گوینده
    
    # َ
        (rf'{ZBAR}', 'a'),
    # ِ
        (rf'{ZIIR}', 'e'),
    # ُ
        (rf'{PISH}', 'o'),
    # ء
        (r'ء', 'ð'),
    # ئ
        (r'ئ', 'ʊ'),
    # أ
        (r'أ', 'ɔ'),
    # إ
        (r'إ', 'θ'),
    #  ّ
        (rf'({CNST}){TSHD}', r'\1\1'),
        (rf'({CNST})({SVOW}){TSHD}', r'\1\1\2'),
        (rf'{TSHD}', r'\1'),

    
    # << ARABIC Form of words in Persian >>
        (rf'({PUNC})({L03})(وا)({L03})({L03})({PUNC})', rf'\1\2{ZBAR}vΛ\4{ZIIR}\5\6'),  # فواعل
        (rf'({PUNC})({L03})(ا)({L03})({L03})({PUNC})', rf'\1\2Λ\4{ZIIR}\5\6'),  # فاعل
        (rf'({PUNC})(م)({L03})({L03})(و)({L03})({PUNC})', rf'\1\2{ZBAR}\3\4\5\6\7'),  # مفعول
        (rf'({PUNC})({L03})({L03})(و)({L03})({PUNC})', rf'\1\2{PISH}\3u\5\6'),  # فعول
        (rf'({PUNC})({L03})({L03})(ا)({L03})({PUNC})', rf'\1\2{ZBAR}\3Λ\5\6'),  # فعال
        (rf'({PUNC})(م)({L03})({L03})({L03})({PUNC})', rf'\1\2{ZBAR}\3\4{ZBAR}\5\6'),  # مفعل
        (rf'({PUNC})(ا)({L03})({L03})(ا)({L03})({PUNC})', rf'\1ą\3\4Λ\6\7'),  # افعال
        (rf'({PUNC})(م)({L03})({L03})({TSHD})({L03})({PUNC})', rf'\1\2{PISH}\3{ZBAR}\4\4{ZBAR}\6\7'),  # مفعّل
        (rf'({PUNC})(ت)({L03})({L03})(ی)({L03})({PUNC})', rf'\1\2{ZBAR}\3\4i\6\7'),  # تفعیل
        (rf'({PUNC})({L03})({L03})(ی)({L03})({PUNC})', rf'\1\2{ZBAR}\3i\5\6'),  # فعیل
        (rf'({PUNC})({L03})({L03})(ی)({L03})(ه)({PUNC})', rf'\1\2{ZBAR}\3i\5E'),  # فعیله
        (rf'({PUNC})(ا)({L03})({L03})({L03})({PUNC})', rf'\1ą\3\4{ZBAR}\5\6'),  # افعل
        (rf'({PUNC})(ت)({L03})(ا)({L03})(ی)({L03})({PUNC})', rf'\1\2{ZBAR}\3Λ\5i\7\8'),  # تفاعیل
        (rf'({PUNC})(ت)({L03})(ا)({L03})({L03})({PUNC})', rf'\1\2{ZBAR}\3Λ\5{PISH}\6\7'),  # تفاعل
        (rf'({PUNC})(ت)({L03})({L03})({L03})({PUNC})', rf'\1\2{ZBAR}\3{ZBAR}\4{PISH}\5\6'),  # تفعل
        (rf'({PUNC})(ت)({L03})({L03})({TSHD})({L03})({PUNC})', rf'\1\2{ZBAR}\3{ZBAR}\4\4{PISH}\6\7'),  # تفعّل
        (rf'({PUNC})(م)({L03})(ا)({L03})({L03})({PUNC})', rf'\1\2{ZBAR}\3Λ\5{ZIIR}\6\7'),  # مفاعل
        (rf'({PUNC})(ا)(س)(ت)({L03})({L03})(ا)({L03})({PUNC})', rf'\1ę\3\4{ZIIR}\5\6Λ\8\9'),  # استفعال
        (rf'({PUNC})(ا)(ن)({L03})({L03})(ا)({L03})({PUNC})', rf'\1ę\3\4{ZIIR}\5Λ\7\8'),  # انفعال
        (rf'({PUNC})(ا)({L03})(ت)({L03})(ا)({L03})({PUNC})', rf'\1ę\3\4{ZIIR}\5Λ\7\8'),  # افتعال
        (rf'({PUNC})(م)(س)(ت)({L03})({L03})({L03})({PUNC})', rf'\1\2{PISH}\3\4{ZBAR}\5\6{ZIIR}\7\8'),  # مستفعل
        (rf'({PUNC})(م)(ت)({L03})(ا)({L03})({L03})({PUNC})', rf'\1\2{PISH}\3{ZBAR}\4Λ\6{ZIIR}\7\8'),  # متفاعل
        (rf'({PUNC})(م)(ت)({L03})({L03})({L03})({PUNC})', rf'\1\2{PISH}\3{ZBAR}\4{ZBAR}\5{ZIIR}\6\7'),  # متفعل
        (rf'({PUNC})(م)(ت)({L03})({L03})({TSHD})({L03})({PUNC})', rf'\1\2{PISH}\3{ZBAR}\4{ZBAR}\5\5{ZIIR}\7\8'),  # متفعّل
        (rf'({PUNC})(م)({L03})(ت)({L03})({L03})({PUNC})', rf'\1\2{PISH}\3\4{ZBAR}\5{ZIIR}\6\7'),  # مفتعل
        (rf'({PUNC})(م)(ن)({L03})({L03})({L03})({PUNC})', rf'\1\2{PISH}\3\4{ZBAR}\5{ZIIR}\6\7'),  # منفعل
        (rf'({PUNC})(م)(ن)({L03})({L03})({L03})(ه)({PUNC})', rf'\1\2{PISH}\3\4{ZBAR}\5{ZIIR}\6E'),  # منفعله

]



# convert (یک ملیون و دویست هزار و سه) to be (یک ملیون O دویست هزار O سه)
PersianNumbers = [
    'یک', 'دو', 'سه', 'چهار', 'پنج', 'شش', 'هفت', 'هشت', 'نه', 'ده', 'یازده', 'دوازده', 'سیزده', 'چهارده', 'پانزده', 'شانزده',
    'هفده', 'هجده', 'نوزده', 'بیست', 'سی', 'چهل', 'پنجاه', 'شصت', 'هفتاد', 'هشتاد', 'نود', 'صد', 'دویست', 'هزار', 'ملیون',
    'میلیارد', 'بیلیون', 'تیلیارد'
]
# Create a regex pattern dynamically
number_pattern = '|'.join(PersianNumbers)  # Join words with |
PersianNumbersPattern = fr'({number_pattern})\s*[\u200c]?\s*و\s*[\u200c]?\s*({number_pattern})'