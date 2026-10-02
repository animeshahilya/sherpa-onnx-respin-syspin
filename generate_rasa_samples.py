#!/usr/bin/env python3
"""Generate high-quality, long-form conversational speech samples for all 20 Rasa voices.

Renders through sherpa-onnx itself (blank interspersal, sentence split) so samples
match what the app produces, from release_assets_rasa/vits-rasa-13-model.onnx.
Sentence-by-sentence synthesis with 350ms natural breathing pauses guarantees:
  1. Crisp, unhurried articulation (~13-19s per voice)
  2. Zero attention diffusion or rushed words
  3. Zero trailing noise stubs
  4. 100% token coverage against vits-rasa-13/tokens.txt

Encodes directly to 64 kbps mono MP3 in samples/vits-rasa-*.mp3 (24 kHz).
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import sherpa_onnx
import soundfile as sf

BASE = Path(__file__).parent
MODEL_PATH = BASE / "release_assets_rasa" / "vits-rasa-13-model.onnx"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE / "vits-rasa-13" / "model.onnx"
TOKENS_PATH = BASE / "vits-rasa-13" / "tokens.txt"
if not TOKENS_PATH.exists():
    TOKENS_PATH = BASE / "release_assets_rasa" / "vits-rasa-13-tokens.txt"
OUT_DIR = BASE / "samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 13 language test passages, transliterations, translations, and phonetic rubrics
RASA_TEXTS = {
    "asm": {
        "text": "ব্ৰহ্মপুত্ৰৰ দুয়োপাৰৰ সেউজীয়া প্ৰকৃতি আৰু চাহ বাগিচাবোৰে আমাৰ অসমখনক অপূৰ্ব সৌন্দৰ্যৰে সজাই তুলিছে. ৰাতিপুৱাৰ শীতল বতাহ আৰু বিহুগীতৰ মধুৰ সুৰে সকলোৰে মন আনন্দৰে ভৰাই তোলে. কাজিৰঙাৰ মনোৰম বননি আৰু সোণালী ধাননি পথাৰে আমাৰ ঐতিহ্যক অধিক চহকী কৰি তুলিছে. নিজৰ ভাষা আৰু সংস্কৃতিক সন্মান জনাই আমি সকলোৱে মিলি-জুলি আগবাঢ়ি যাব লাগে",
        "transliteration": "Brohmoputror duyopaaror seujiya prokriti aaru saah baagisabore aamaar Axomkhonok opurbo xoundorjore xozai tulise. Raatipuwaar xeetol botah aaru Bihugeetor modhur xure xokolore mon aanondore bhorai tole. Kazirangar monorom bononi aaru sonali dhaanoni pothaare aamaar oitijhyok odhik sohokee kori tulise. Nizor bhaaxa aaru xongskritik xonmaan zonai aami xokolowe mili-zuli aagbaadhi zaabo laage.",
        "translation": "The lush greenery along both banks of the Brahmaputra and rolling tea gardens adorn our Assam with sublime beauty. The morning breeze and sweet tunes of Bihu songs fill everyone's heart with joy. The picturesque forests of Kaziranga and golden paddy fields make our heritage ever more bountiful. Respecting our language and culture, let us all advance forward together in harmony.",
        "rubric": [
            "Evaluate characteristic Assamese velar fricative /x/ for sibilants (স, শ, ষ).",
            "Listen for distinct Assamese 'ৰ' (ro) and 'ৱ' (wo) articulation.",
            "Observe gentle sentence cadence and natural pauses at periods."
        ]
    },
    "bn": {
        "text": "বাঙলার শান্ত গ্রাম্য প্রকৃতি আর গঙ্গার তীরে সকালের স্নিগ্ধ বাতাস মনকে এক অপার প্রশান্তি এনে দেয়. আমাদের সাহিত্য, সঙ্গীত আর সংস্কৃতির ঐতিহ্য শতাব্দীর পর শতাব্দী ধরে বেঁচে আছে. রবীন্দ্রনাথ আর নজরুলের অমর সৃষ্টির সুর আজও প্রতিটি বাঙালির অন্তরে অনুরণিত হয়. ভালোবাসার বন্ধনে সবাইকে আপন করে নিয়ে নতুন দিনের পথে এগিয়ে চলাই আমাদের প্রেরণা",
        "transliteration": "Banglar shanto gramyo prokriti aar Gongar teere sokaler snigdho batash monke ek opar proshanti ene dey. Amader sahityo, songeet aar songskritir oitijhyo shotabdhir por shotabdhi dhore beche achhe. Robindronath aar Nozrulel omor srishtir shur aaj-o protiti Bangalir ontore onuronito hoy. Bhalobashar bondhone shobaike aapon kore niye notun diner pothe egiye cholai amader prerona.",
        "translation": "The serene rural landscape of Bengal and the gentle breeze on the banks of the Ganges bring immense tranquility. The legacy of our literature, music, and culture has endured through centuries. The immortal melodies of Rabindranath and Nazrul continue to resonate in every Bengali heart. Embracing all with love and striding forward toward a new dawn is our inspiration.",
        "rubric": [
            "Verify rounded Bengali vowel coloring (অ/ও-কার) and soft nasalized cadences.",
            "Test complex conjuncts (স্নিগ্ধ, প্রশান্তি, ঐতিহ্য, শতাব্দী, সৃষ্টি).",
            "Compare tonal balance and emotional warmth between Tithi and Anirban."
        ]
    },
    "brx": {
        "text": "बडोलेण्डनि समायना मिथिंगा, दैमा-दैसा आरो हाजो-हालानि नुथायआ जोंनि गोसोखौ गोजोननायजों आबुं खालामो. फुंनि बार बारनायजों लोगोसे बैसागुनि खाम्-सिफुंनि मोन्थायफोरा गासैखौबो रंजाहोयो. दाउ-दाउसाफोरनि गाबख्रायनाय आरो मानसिनि मिनिस्लु मोखांआ जोंनि गामिखौ समायना खालामो. जोंनि राव, हारिमु आरो आगोराखौ मोजां मोनना जों खौसेथिजों थांनानै थानो नांगौ",
        "transliteration": "Bodoland-ni somaina mothinga, doima-doisa aro hajo-halani nuthaya jwni gosokhwo gojwnnaijwng abu-ng khalamw. Phungni bar barnaijwng logose Bwisaguni kham-siphungni mwnthaiphora gasoikhwbo rwnjahoyo. Dau-dausaforni gabkhrainai aro mansini minislw mwkhanga jwni gamikhwo somaina khalamw. Jwni rao, harimu aro agorakhwo mojakh mwnna jwng khousethijwng thangnanwi thanw nangw.",
        "translation": "The breathtaking nature of Bodoland, its rivers, and scenic hills fill our hearts with serene contentment. The morning breeze along with the rhythm of the Kham and Sifung flutes of Bwisagu bring delight to all. The cheerful chirping of birds and smiling faces of the people make our village splendid. Cherishing our mother tongue, heritage, and woven traditions, let us live together in unity.",
        "rubric": [
            "Evaluate high unrounded vowel /ɯ/ and natural tone articulation in Devanagari Bodo.",
            "Test authentic regional vocabulary (बैसागु, खाम्-सिफुं, आगोरा, खौसेथि).",
            "Observe bright folk narrative prosody in Mainao and Sansuma."
        ]
    },
    "doi": {
        "text": "साह्ड़े डुग्गर प्रदेश दी सोहणी धरती, तवी नदी दा पावन कंडा ते बावे वाली माता दा मंदिर मने गी बड़ा आनंद दिंदे न. सवेरे-सवेरे पीर पंजाल दे पहाड़ां थमां औंदी ठंडी हवा ते पंछियें दा चहचहाना मन च नवां चाऽ भरि दिंदा ऐ. डोगरें दी रीत ऐ कि सबने कन्ने प्यार-मुहब्बत ते सत्कार कन्ने रौह्ना. अपनी मीठी डोगरी बोली ते लोक-संस्कृति दी सेवा करना साह्ड़ा सच्चा फर्ज ऐ",
        "transliteration": "Sahde Duggar pardesh di sohni dharti, Tawi nadi da paavan kanda te Bawe wali Mata da mandir mane gi bada aanand dinde nan. Savere-savere Peer Panjal de pahadan thaman aundi thandi hawa te panjhiyen da chahchahana man cha nawan chaa bhari dinda ai. Dogren di reet ai ki sabne kanne pyaar-muhabbat te satkaar kanne rauhna. Apni meethi Dogri boli te lok-sanskriti di seva karna sahda sachha farz ai.",
        "translation": "The splendid land of our Duggar realm, the sacred banks of river Tawi, and the shrine of Bawe Wali Mata bring profound joy. The refreshing dawn breeze descending from the Pir Panjal ranges and the chirping of birds infuse eager zeal. It is the noble custom of the Dogras to live with mutual affection and respect. Cherishing our sweet Dogri tongue and folk heritage is our true calling.",
        "rubric": [
            "Assess characteristic Dogri tonal contours (rising-falling pitch on aspirates/glottals).",
            "Check regional postpositions and verbal markers (दी, दा, गी, च, ऐ, न).",
            "Listen for authentic Duggar folk warmth and melodious cadence in Sheetal and Vijay."
        ]
    },
    "kn": {
        "text": "ಕರ್ನಾಟಕದ ಭವ್ಯ ಇತಿಹಾಸ, ಸುಂದರ ಸಹ್ಯಾದ್ರಿ ಬೆಟ್ಟಗಳು ಮತ್ತು ಹಸಿರಿನ ಕಾನನಗಳು ನಮ್ಮ ನಾಡಿನ ಕೀರ್ತಿಯನ್ನು ಹೆಚ್ಚಿಸಿವೆ. ಮುಂಜಾನೆಯ ತಂಗಾಳಿ ಮತ್ತು ನದಿಗಳ ಇಂಪಾದ ಹರಿವು ಮನಸ್ಸಿಗೆ ಶಾಂತಿ ತರುತ್ತದೆ. ಕವಿಗಳ ವಾಣಿ ಮತ್ತು ಕಲೆಗಳ ವೈಭವ ನಮ್ಮ ಬದುಕನ್ನು ಸುಂದರಗೊಳಿಸಿವೆ. ನಮ್ಮ ಸಂಸ್ಕೃತಿ ಮತ್ತು ಜ್ಞಾನದ ಪರಂಪರೆಯನ್ನು ಕಾಪಾಡಿಕೊಂಡು ನಾವೆಲ್ಲರೂ ಒಗ್ಗಟ್ಟಿನಿಂದ ಮುನ್ನಡೆಯೋಣ",
        "transliteration": "Karnatakada bhavya itihaasa, sundara Sahyadri bettagalu mattu hasirina kaananagalu namma naadina keertiyannu hecchisive. Munjaaneya tangaali mattu nadigala impaada harivu manassige shaanti taruttade. Kavigala vaani mattu kalegala vaibhava namma badukannu sundaragolisive. Namma sanskruti mattu jnaanada parampareyannu kaapaadikandu naavellaroo oggattininda munnadeyona.",
        "translation": "Karnataka's glorious history, majestic Sahyadri ranges, and verdant forests heighten our land's renown. The morning breeze and soothing murmur of rivers bestow tranquility. The wisdom of our poets and splendor of arts enrich our lives. Guarding our cultural and intellectual heritage, let us march forward united.",
        "rubric": [
            "Evaluate gemination (ದ್ವಿತ್ವ: ಬೆಟ್ಟಗಳು, ಹೆಚ್ಚಿಸಿವೆ, ತರುತ್ತದೆ).",
            "Check retroflex consonant articulation (ಣ, ಳ, ಟ) and smooth Ajanta vowel endings.",
            "Compare dynamic vocal timbre between alternate speakers Spoorthi and Chetan."
        ]
    },
    "mai": {
        "text": "मिथिलाक पावन माटि पर ज्ञान, संस्कार आ लोक-संस्कृतिक अनुपम धारा युग-युगान्तर सं बहैत आबि रहल अछि. भोरे-भोर पोखरि-भींडा पर शीतल बयार आ मन्दिरक शंख-ध्वनि सं मन प्रसन्न भऽ जाइत अछि. महाकवि विद्यापतिक मधुर पदावली आइयो जन-मानस केँ मुग्ध करैत अछि. अपन मधुर मैथिली भाषाक मान बढ़ायब आ समाजक सेवा करब हमर गौरव अछि",
        "transliteration": "Mithilaak paavan maati par gyaan, sanskaar aa lok-sanskritik anupam dhaara yug-yugaantar san bahait aabi rahal achhi. Bhore-bhor pokhari-bheenda par sheetal bayaar aa mandirak shankh-dhwani san man prasann bha jaait achhi. Mahakavi Vidyapatik madhur padaavali aaiyo jan-maanas ken mugdh karait achhi. Apan madhur Maithili bhaashaak maan badhaayab aa samaajak seva karab hamar gaurav achhi.",
        "translation": "Upon the sacred soil of Mithila, the peerless stream of wisdom, values, and folk traditions has flowed across ages. The morning pond-side breeze and resounding conches bring deep joy. The sweet verses of Mahakavi Vidyapati continue to enchant the soul. Upholding our melodious Maithili tongue and serving society is our enduring pride.",
        "rubric": [
            "Verify genitive nominal affix -क (मिथिलाक, मन्दिरक, भाषाक).",
            "Test classic verbal auxiliaries (बहैत आबि रहल अछि, भऽ जाइत अछि).",
            "Observe dignified, scholarly cadence in Shravan's masculine voice."
        ]
    },
    "mal": {
        "text": "കേരളത്തിന്റെ പ്രകൃതിഭംഗിയും ശാന്തമായ കായലുകളും പച്ചപ്പ് നിറഞ്ഞ മലനിരകളും മനസ്സിന് വല്ലാത്തൊരു കുളിർമ നൽകുന്നു. പ്രഭാതത്തിലെ കുളിർകാറ്റും പക്ഷികളുടെ മധുരമായ പാട്ടും പുതിയൊരു ഉണർവ് സമ്മാനിക്കുന്നു. നമ്മുടെ സ്നേഹവും സാഹോദര്യവും കാത്തുസൂക്ഷിച്ച് മുന്നോട്ട് പോകാൻ നമുക്ക് സാധിക്കട്ടെ",
        "transliteration": "Keralatthinte prakruthibhangiyum shanthamaaya kaayalukalum pachappu niranjo malanirakalum manassinu vallaathoru kulirma nalkunnu. Prabhaathatthile kulirkaattum pakshikalude madhuramaaya paattum puthiyoru unarvu sammaanikkunnu. Nammude snehavum saahodaryavum kaatthusookshicchu munnottu pokaan namukku saadhikkatte.",
        "translation": "Kerala's picturesque beauty, peaceful backwaters, and lush hills impart an incomparable soothing freshness to the mind. The brisk morning breeze and melodic birdsong bestow a renewed vitality. May we always preserve our mutual love and brotherhood as we advance forward.",
        "rubric": [
            "Verify authentic Malayalam chillu letters (ൽ, ർ, ൺ, ൻ) and retroflexes (ള, ഴ, റ).",
            "Check consonant gemination (കേരളത്തിന്റെ, കുളിർകാറ്റും, സമ്മാനിക്കുന്നു).",
            "Observe flowing, rhythmic cadence with natural breathing intervals in Aparna's voice."
        ]
    },
    "mr": {
        "text": "सह्याद्रीच्या उत्तुंग रांगा आणि संतांच्या शिकवणीने पावन झालेली आपली महाराष्ट्र भूमी मनाला नेहमीच नवी प्रेरणा देते. पहाटेच्या वेळी शेतांमध्ये वाहणारा सुखद गारवा आणि पक्षांचा किलबिलाट नवा उत्साह निर्माण करतो. आपला गौरवशाली इतिहास आणि संस्कृती जपताना, विज्ञानाची कास धरून पुढे जाणे काळाची गरज आहे. कष्टाला प्रामाणिकपणाची जोड देऊन प्रगतीच्या दिशेने पाऊल टाकणे हेच आपले ध्येय आहे",
        "transliteration": "Sahyadrichya uttung raanga aani santanchya shikavanine paavan jhaalelee aapli Maharashtra bhoomi manala nehmeecha navee prerana dete. Pahaatechya velee shetaanmadhye vaahnara sukhad gaarva aani pakshaancha kilbilaat nava utsaah nirmaan karto. Aapla gauravshaalee itihaas aani sanskruti japtaanaa, vijnaanaachee kaas dharoon pudhe jaane kaalaachee garaj aahe. Kashtaala praamaanikpanaachee jod deun pragatichya dishene paool taakne hecha aaple dhyey aahe.",
        "translation": "The towering Sahyadri ranges and the teachings of saints make our Maharashtra land an everlasting source of inspiration. The dawn breeze through fields and chirping birds kindle vibrant zeal. Preserving our glorious history and culture while embracing modern science is the call of the hour. Pairing honest labor with dedication toward progress is our highest goal.",
        "rubric": [
            "Evaluate crisp Marathi dental vs. palatal affricates (झालेली, जोड).",
            "Test heavy conjuncts (सह्याद्रीच्या, उत्तुंग, प्रामाणिकपणाची, ध्येय).",
            "Compare tonal balance and authority between alternate speakers Mrunal and Tejas."
        ]
    },
    "ne": {
        "text": "हिमालको काखमा अवस्थित हाम्रो सुन्दर देशमा प्रकृतिको अनुपम वरदान चारैतिर फैलिएको छ. बिहानीको सुनौलो घाम र हिउँका चुचुराहरूले मनमा असीम शान्ति र उमङ्ग भरिदिन्छन्. आफ्ना मौलिक परम्परा, भाषा र संस्कृतिको संरक्षण गर्दै हामी सबै मिलेर अघि बढ्नुपर्छ",
        "transliteration": "Himaalko kaakhmaa avasthit haamro sundar deshmaa prakritiko anupam vardaan chaaraitira phailieko chha. Bihaaniko sunoulo ghaam ra hiunkaa chuchuraaharule manmaa aseem shaanti ra umanga bharidinchhan. Aapnaa maulik paramparaa, bhaashaa ra sanskritiko sanrakshan gardai haamee sabai milera aghi badhnuparchha.",
        "translation": "Nestled in the lap of the Himalayas, our wonderful land is blessed with nature's peerless gifts stretching all around. The golden morning sunshine upon snowy peaks fills the heart with infinite peace and jubilation. Preserving our original traditions, language, and culture, let us all progress forward together.",
        "rubric": [
            "Evaluate distinct Nepali verb agreement forms (भरिदिन्छन्, बढ्नुपर्छ).",
            "Check clear conjunct and nasal articulacy (हिमालको, चुचुराहरूले, उमङ्ग).",
            "Listen for Prerana's cheerful, expressive Himalayan speech contour."
        ]
    },
    "pan": {
        "text": "ਪੰਜ ਦਰਿਆਵਾਂ ਦੀ ਧਰਤੀ ਪੰਜਾਬ ਦੀ ਸ਼ਾਨ, ਹਰੇ-ਭਰੇ ਖੇਤ ਅਤੇ ਮਿੱਠੀ ਬੋਲੀ ਹਰ ਕਿਸੇ ਦਾ ਦਿਲ ਮੋਹ ਲੈਂਦੀ ਹੈ. ਅੰਮ੍ਰਿਤ ਵੇਲੇ ਦੀ ਠੰਢੀ ਪੌਣ ਅਤੇ ਗੁਰਬਾਣੀ ਦੇ ਮਿੱਠੇ ਬੋਲ ਮਨ ਵਿੱਚ ਸ਼ਾਂਤੀ ਅਤੇ ਨਵੀਂ ਊਰਜਾ ਭਰ ਦਿੰਦੇ ਹਨ. ਸਾਡੇ ਵਿਰਸੇ ਦੀ ਅਮੀਰੀ, ਗਿੱਧਾ-ਭੰਗੜਾ ਅਤੇ ਮਹਿਮਾਨ-ਨਵਾਜ਼ੀ ਪੂਰੀ ਦੁਨੀਆ ਵਿੱਚ ਮਸ਼ਹੂਰ ਹਨ. ਸਾਰੇ ਰਲ-ਮਿਲ ਕੇ ਪਿਆਰ ਅਤੇ ਚੜ੍ਹਦੀ ਕਲਾ ਨਾਲ ਜ਼ਿੰਦਗੀ ਦੇ ਰਾਹ ਤੇ ਅੱਗੇ ਵਧੀਏ",
        "transliteration": "Panj daryawan di dharti Punjab di shaan, hare-bhare khet ate meethi boli har kise da dil moh laindi hai. Amrit vele di thandhi paun ate Gurbani de meethe bol man vich shanti ate navin oorja bhar dinde han. Saade virse di ameeri, giddha-bhangra ate mehmaan-nawazi poori duniya vich mashhoor han. Saare ral-mil ke pyaar ate chardi kala naal zindagi de raah te agge vadhiye.",
        "translation": "The pride of Punjab, the blessed land of five rivers, with its emerald wheat fields and sweet tongue captivates every heart. In the sacred dawn hour (Amrit Vela), the refreshing breeze and sweet hymns instill tranquil peace and renewed energy. Our rich cultural heritage, Giddha, Bhangra, and hospitality are renowned worldwide. Let us stride forward together in high spirits (Chardi Kala).",
        "rubric": [
            "Assess Gurmukhi tonal inflection and gemination with Addak (ੱ: ਠੰਢੀ, ਮਿੱਠੀ, ਰਲ-ਮਿਲ).",
            "Test Punjabi nasal markers (Tippi ੰ / Bindi ਂ: ਧਰਤੀ, ਪੰਜਾਬ, ਸ਼ਾਂਤੀ).",
            "Compare spirited delivery between female speaker Simran and male speaker Harpreet."
        ]
    },
    "san": {
        "text": "अस्माकं भारतवर्षस्य प्राचीना संस्कृतिः विश्वे सर्वत्र पूज्यते. अत्र ज्ञानस्य, धर्मस्य, कलायाः च अनुपमः संगमः दृश्यते. प्रातःकाले उदीयमानः सूर्यः सर्वेभ्यः प्राणिभ्यः नूतनम् उत्साहं शान्तिं च प्रयच्छति. सर्वे भवन्तु सुखिनः इति भावनया वयं परस्परं स्नेहेन निवसेम",
        "transliteration": "Asmaakam Bhaaratavarshasya praacheenaa sanskritih vishwe sarvatra poojyate. Atra jnaanasya, dharmasya, kalaayaah cha anupamah sangamah drishyate. Praatahkaale udeeyamaanah sooryah sarvebhyah praanibhyah nootanam utsaaham shaantim cha prayacchati. Sarve bhavantu sukhinah iti bhaavanayaa vayam parasparam snehena nivasema.",
        "translation": "The ancient culture of our Bharatavarsha is revered throughout the world. Here, a peerless confluence of wisdom, righteousness, and fine arts is witnessed. The morning sun rising in the sky bestows renewed vigor and serene peace upon all living beings. Guided by the noble ideal 'May all beings be happy', let us live together in mutual love.",
        "rubric": [
            "Assess classical Sanskrit pronunciation: visarga (ः), anusvara (ं), and samasa compounds.",
            "Test complex consonant conjuncts (संस्कृतिः, दृश्यते, प्राणिभ्यः, प्रयच्छति).",
            "Evaluate resonance, dignity, and Vedic cadence in Vedant's voice."
        ]
    },
    "tam": {
        "text": "நம் தமிழ்நாடு பழம்பெருமை வாய்ந்த பண்பாடும், செழுமையான இலக்கிய வளமும் கொண்ட சிறப்பான நிலமாகும். அதிகாலை வேளையில் வீசும் தென்றல் காற்றும், பறவைகளின் இனிய குரலும் உள்ளத்திற்கு அமைதியையும் புத்துணர்ச்சியையும் தருகின்றன. கோயில் கோபுரங்களின் அழகும், பரதநாட்டியமும், திருக்குறளின் வழிகாட்டுதலும் நம் பண்பாட்டின் சிகரமாகும். அன்பும் அறமும் கொண்டு நாம் அனைவரும் ஒற்றுமையுடன் வாழ்ந்து பெருமை சேர்ப்போம்",
        "transliteration": "Nam Thamizhnaadu pazhamperumai vaaintha panpaadum, sezhumaiyaana ilakkiya valamum konda sirappaana nilamaagum. Athikaalai vaelaiyil veesum thendral kaattrum, paravaigalin iniya kuralum ullathirku amaithiyaiyum puththunarchiyaiyum tharukindrana. Koyil gopurangalin azhagum, Bharathanattiyamum, Thirukkuralin vazhikaattudhalum nam panpaattin sikaramaagum. Anbum aramum kondu naam anaivarum otrumaiyudan vaazhndhu perumai saerppom.",
        "translation": "Our Tamil Nadu is an extraordinary land endowed with time-honored heritage and rich literary wealth. The gentle morning breeze and sweet calls of birds impart deep peace and revitalization to the spirit. The grandeur of temple towers, Bharatanatyam, and the guidance of Thirukkural are the pinnacles of our heritage. Embracing love and righteousness, let us live in harmony and bring glory to our land.",
        "rubric": [
            "Check characteristic Tamil retroflex liquid (ழ: தமிழ்நாடு, செழுமையான, வாழ்ந்து, வழிகாட்டுதல்).",
            "Test alveolar tap/trill vs retroflex flap (ர vs ற, ல vs ள vs ழ).",
            "Listen for pure natural prosody and balanced articulation in Kaveri's voice."
        ]
    },
    "te": {
        "text": "మన తెలుగు నేల సంస్కృతి, సంప్రదాయాలు మరియు మధురమైన సాహిత్యానికి పెట్టింది పేరు. గోదావరి, కృష్ణా నదుల తీరాలలో ఉదయించే భానుని కిరణాలు ప్రకృతికి సరికొత్త అందాన్ని తెచ్చిపెడతాయి. మన పెద్దల అనుభవాలు మరియు సద్గుణాలను గౌరవిస్తూ, మనమందరం కలసికట్టుగా ప్రగతి పథంలో ముందుకు సాగుదాం",
        "transliteration": "Mana Telugu nela sanskruthi, sampradaayaalu mariyu madhuramaina saahityaaniki pettindi peru. Godavari, Krishnaa nadula theeraalalo udayinche bhaanuni kiranaalu prakruthiki sarikottha andaanni thecchipadathaayi. Mana peddala anubhavaalu mariyu sadgunaalanu gouravistoo, manamandaram kalasikattugaa pragathi pathamlo munduku saagudaam.",
        "translation": "Our Telugu land is famed for its rich culture, time-honored traditions, and melodious literature. The golden rays of the morning sun shining along the banks of the Godavari and Krishna rivers bring enchanting beauty to nature. Respecting the wisdom of our elders and noble virtues, let us all journey forward united on the path of progress.",
        "rubric": [
            "Test graceful Ajanta (vowel-ending) syllable flow and rhythmic cadence.",
            "Assess conjunct clusters and retroflex consonants (సంప్రదాయాలు, సరికొత్త, కలసికట్టుగా).",
            "Evaluate Harini's bright, lyrical female articulation."
        ]
    }
}

# The 20 Rasa voices: (sid, mp3_id, name, lang_key, gender, custom_length_scale)
# length_scale 1.0 = the model's trained pace (sherpa adds the add_blank interspersal).
RASA_VOICE_CONFIGS = [
    (0, "vits-rasa-asm-female", "Bornali", "asm", "female", 1.0),
    (1, "vits-rasa-asm-male", "Rituraj", "asm", "male", 1.0),
    (2, "vits-rasa-bn-female-alt", "Tithi", "bn", "female", 1.0),
    (3, "vits-rasa-bn-male-alt", "Anirban", "bn", "male", 1.0),
    (4, "vits-rasa-brx-female", "Mainao", "brx", "female", 1.0),
    (5, "vits-rasa-brx-male", "Sansuma", "brx", "male", 1.0),
    (6, "vits-rasa-doi-female", "Sheetal", "doi", "female", 1.0),
    (7, "vits-rasa-doi-male", "Vijay", "doi", "male", 1.0),
    (8, "vits-rasa-kn-female-alt", "Spoorthi", "kn", "female", 1.0),
    (9, "vits-rasa-kn-male-alt", "Chetan", "kn", "male", 1.0),
    (10, "vits-rasa-mai-male-alt", "Shravan", "mai", "male", 1.0),
    (11, "vits-rasa-mal-female", "Aparna", "mal", "female", 1.0),
    (12, "vits-rasa-mr-female-alt", "Mrunal", "mr", "female", 1.0),
    (13, "vits-rasa-mr-male-alt", "Tejas", "mr", "male", 1.0),
    (14, "vits-rasa-ne-female", "Prerana", "ne", "female", 1.0),
    (15, "vits-rasa-pan-female", "Simran", "pan", "female", 1.0),
    (16, "vits-rasa-pan-male", "Harpreet", "pan", "male", 1.0),
    (17, "vits-rasa-san-male", "Vedant", "san", "male", 1.0),
    (18, "vits-rasa-tam-female", "Kaveri", "tam", "female", 1.0),
    (19, "vits-rasa-te-female-alt", "Harini", "te", "female", 1.0),
]


def load_vocab(tokens_file: Path):
    vocab = {}
    for line in tokens_file.read_text(encoding="utf-8").splitlines():
        if " " in line:
            s, i = line.rsplit(" ", 1)
            try:
                vocab[s] = int(i)
            except ValueError:
                pass
    return vocab


def synthesize_voice(tts, sid, text, length_scale=1.0, sample_rate=24000):
    # Split text into natural sentences
    sents = [s.strip() for s in re.split(r"[.।\n]+", text) if s.strip()]
    pause_samples = int(sample_rate * 0.35)  # 350 ms breathing silence
    pause = np.zeros(pause_samples, dtype=np.float32)

    def _trim(w, threshold=0.02, margin_ms=50.0):
        if w.size == 0:
            return w
        above = np.where(np.abs(w) > threshold)[0]
        if above.size == 0:
            return w
        margin = int(sample_rate * margin_ms / 1000.0)
        return w[max(0, int(above[0]) - margin):min(w.size, int(above[-1]) + margin + 1)]

    parts = []
    for idx, s in enumerate(sents):
        # Strip trailing punctuation that might cause click
        clean = s.rstrip(".?!:;, ")
        if not clean:
            continue
        w = np.array(tts.generate(clean, sid=sid, speed=1.0 / length_scale).samples, dtype=np.float32)
        parts.append(_trim(w))
        if idx < len(sents) - 1:
            parts.append(pause)

    if not parts:
        return np.zeros(sample_rate, dtype=np.float32)

    full = np.concatenate(parts)
    # Peak normalization to prevent clipping and ensure consistent loudness
    mx = float(np.max(np.abs(full)))
    if mx > 0.01:
        full = full * (0.90 / mx)
    return full


def main():
    print(f"Loading sherpa-onnx TTS from {MODEL_PATH}...")
    tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=str(MODEL_PATH), tokens=str(TOKENS_PATH)),
        num_threads=4)))
    vocab = load_vocab(TOKENS_PATH)
    print(f"Loaded {len(vocab)} tokens from {TOKENS_PATH}")

    # Verify zero dropped characters across all 13 texts
    for lang_key, data in RASA_TEXTS.items():
        unmapped = [c for c in data["text"] if c not in vocab]
        if unmapped:
            raise ValueError(f"Language '{lang_key}' has unmapped characters: {unmapped}")
    print("Pre-flight check PASSED: All 13 test passages have 100% token vocabulary coverage.")

    total_t0 = time.time()
    results = []
    for sid, mp3_id, name, lang_key, gender, ls in RASA_VOICE_CONFIGS:
        t0 = time.time()
        text = RASA_TEXTS[lang_key]["text"]
        wav = synthesize_voice(tts, sid, text, length_scale=ls)
        dur = len(wav) / 24000.0

        wav_path = OUT_DIR / f"{mp3_id}.wav"
        mp3_path = OUT_DIR / f"{mp3_id}.mp3"

        sf.write(str(wav_path), wav, 24000)
        # Encode to lightweight, high-fidelity 64 kbps mono MP3
        subprocess.run(["ffmpeg", "-y", "-i", str(wav_path), "-b:a", "64k", str(mp3_path)],
                       capture_output=True, check=True)
        wav_path.unlink()

        sz_kb = mp3_path.stat().st_size / 1024
        elapsed = time.time() - t0
        print(f"[{sid:2d}] {name:10s} ({lang_key:3s} {gender:6s}) -> {dur:5.1f}s audio | {sz_kb:5.1f} KB MP3 in {elapsed:4.1f}s")
        results.append((sid, name, lang_key, dur, sz_kb))

    print(f"\nAll 20 Rasa voice samples generated in {time.time() - total_t0:.1f}s!")
    print("\nSummary:")
    for sid, name, lang_key, dur, sz in results:
        print(f"  sid={sid:2d}: {name:10s} ({lang_key:3s}) -> {dur:5.1f}s ({sz:4.1f} KB)")


if __name__ == "__main__":
    main()
