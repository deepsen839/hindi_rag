# Six mandatory questions — source-verified reference answers

These are **human-verified reference answers transcribed from the supplied PDF**, not fabricated model-run logs. Execute `python -m evaluation.run_assignment_queries` after building the Chroma index and loading Qwen to produce actual runtime answers in `reports/assignment_answers.md` and `.json`.

1. **SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?**  
   उत्तर: SLV-III ने जुलाई 1980 में रोहिणी उपग्रह को कक्षा में स्थापित किया।  
   स्रोत: PDF पृष्ठ 8 · अनुभाग `इसरो के वर्षष और SLV-III` · chunk `3d96c2ef13855bab`.

2. **कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?**  
   उत्तर: 1997 में।  
   स्रोत: PDF पृष्ठ 16 · अनुभाग `पुरस्काार एवं सम्माान` · chunk `58bd111b23086b0f` (तालिका-पंक्ति chunk `b223f13989dd02a7` भी उपलब्ध है).

3. **पोखरण-II में कलाम की क्या भूमिका थी?**  
   उत्तर: वे उस समय रक्षा मंत्री के वैज्ञानिक सलाहकार और DRDO के प्रमुख थे। उन्होंने परमाणु ऊर्जा विभाग के वैज्ञानिकों के साथ मिलकर केंद्रीय संगठनात्मक भूमिका निभाई।  
   स्रोत: PDF पृष्ठ 10 · अनुभाग `पोखरण-II: 1998 के परमाणु परीक्षण` · chunk `34f9ad59201283ff`.

4. **Which institution did Kalam attend to study aeronautical engineering?**  
   उत्तर: चेन्नई का मद्रास इंस्टीट्यूट ऑफ टेक्नोलॉजी (MIT)।  
   स्रोत: PDF पृष्ठ 6 · अनुभाग `कॉलेज के वर्षष और उड़ाान का आकर्षषण` · chunk `f70256eda3065fa9`.

5. **Who co-wrote the autobiography, and in what year was it published?**  
   उत्तर: वैज्ञानिक अरुण तिवारी ने इसे कलाम के साथ लिखा; यह 1999 में प्रकाशित हुई।  
   स्रोत: PDF पृष्ठ 3 · अनुभाग `ए.पी.जे. अब्दुल कलाम कौन थे?` · chunk `3e3aa45e49f2bbdd`.

6. **How and where did Kalam die in 2015?**  
   उत्तर: 27 जुलाई 2015 को शिलांग के भारतीय प्रबंधन संस्थान में विद्यार्थियों को व्याख्यान देते समय वे गिर पड़े और कुछ ही देर बाद उनका निधन हो गया।  
   स्रोत: PDF पृष्ठ 15 · अनुभाग `अंतिम क्षण तक शिक्षक (2007–2015)` · chunk `9e765d55a704207d`.

**Citation note:** Page numbers are 1-based physical PDF pages. Chunk IDs are generated from the normalized source text and ingestion metadata; rerunning ingestion with changed chunking settings may change them.
