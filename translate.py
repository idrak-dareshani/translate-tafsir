#!/usr/bin/env python3
"""
Multi-Language Tafsir Translator
A Python script to translate Arabic or Urdu Quranic commentary (tafsir) to English
Automatically detects source language and translates using deep-translator
"""

import time
import re
import logging
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import json
from datetime import datetime

# Deep-translator imports
from deep_translator import GoogleTranslator, single_detection, batch_detection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TafsirTranslator:
    def __init__(self, delay_between_requests: float = 1.0):
        """
        Initialize the multi-language tafsir translator
        
        Args:
            delay_between_requests: Delay between API calls to respect rate limits
        """
        self.delay_between_requests = delay_between_requests
        self.target_lang = 'en'  # English
        self.supported_languages = {
            'ar': 'Arabic',
            'ur': 'Urdu'
        }
        
        logger.info("Multi-language Tafsir Translator initialized")
    
    def detect_language(self, text: str, confidence_threshold: float = 0.8) -> Tuple[str, float, str]:
        """
        Detect the source language of the text
        
        Returns:
            Tuple of (language_code, confidence, language_name)
        """
        try:
            # Clean text for better detection
            clean_text = self._clean_text_for_detection(text)
            
            # Use single detection for better accuracy
            detected_lang = single_detection(clean_text, api_key=None)
            
            # Map detected language
            if detected_lang in self.supported_languages:
                lang_name = self.supported_languages[detected_lang]
                confidence = 0.9  # High confidence for supported languages
            else:
                # Default to Arabic for unknown languages with Arabic script
                if self._has_arabic_script(clean_text):
                    detected_lang = 'ar'
                    lang_name = 'Arabic'
                    confidence = 0.6
                elif self._has_urdu_script(clean_text):
                    detected_lang = 'ur'
                    lang_name = 'Urdu'
                    confidence = 0.6
                else:
                    detected_lang = 'ar'  # Default fallback
                    lang_name = 'Arabic (fallback)'
                    confidence = 0.3
            
            logger.info(f"Detected language: {lang_name} ({detected_lang}) with confidence: {confidence:.2f}")
            return detected_lang, confidence, lang_name
            
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}. Defaulting to Arabic.")
            return 'ar', 0.5, 'Arabic (default)'
    
    def _clean_text_for_detection(self, text: str) -> str:
        """Clean text to improve language detection accuracy"""
        # Take a sample from the middle of the text for better detection
        text_length = len(text)
        if text_length > 1000:
            start = text_length // 4
            end = start + 500
            text = text[start:end]
        
        # Remove excessive punctuation and numbers
        text = re.sub(r'[0-9]+', '', text)
        text = re.sub(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', ' ', text)
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def _has_arabic_script(self, text: str) -> bool:
        """Check if text contains Arabic script characters"""
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
        arabic_chars = len(arabic_pattern.findall(text))
        total_chars = len(re.findall(r'\w', text))
        return arabic_chars > total_chars * 0.3 if total_chars > 0 else False
    
    def _has_urdu_script(self, text: str) -> bool:
        """Check if text contains Urdu-specific characters"""
        # Urdu uses additional characters beyond basic Arabic
        urdu_specific = re.compile(r'[\u0679\u067E\u0686\u0688\u0691\u06BA\u06BE\u06C1\u06C3\u06CC\u06D2]')
        return len(urdu_specific.findall(text)) > 0
    
    def preprocess_text(self, text: str, language: str) -> str:
        """
        Preprocess text based on detected language
        """
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text.strip())
        
        if language in ['ar', 'ur']:
            # Handle Arabic/Urdu/Persian punctuation
            text = re.sub(r'[«»]', '"', text)
            text = re.sub(r'[،]', ',', text)
            text = re.sub(r'[؛]', ';', text)
            text = re.sub(r'[؟]', '?', text)
            
            # Handle Quranic verse markers
            text = re.sub(r'[﴾﴿]', '', text)
            text = re.sub(r'[۞]', '', text)  # Quranic markers
            
            # Convert Arabic-Indic numerals to Western numerals
            arabic_nums = '٠١٢٣٤٥٦٧٨٩'
            english_nums = '0123456789'
            for ar_num, en_num in zip(arabic_nums, english_nums):
                text = text.replace(ar_num, en_num)
                
            # Handle Urdu-specific formatting
            if language == 'ur':
                # Remove excessive Urdu punctuation
                text = re.sub(r'[۔]+', '۔', text)  # Urdu full stop
                
        return text
    
    def split_text_intelligently(self, text: str, language: str, max_length: int = 3000) -> List[str]:
        """
        Split text into chunks prioritizing newlines, then by words, respecting max_length.
        
        Args:
            text: The input text to split.
            language: The detected language (used for sentence patterns, if applicable).
            max_length: The maximum desired length for each chunk.
                        Note: Google Translate's recommended max is 5000 chars,
                        with advanced supporting up to 30000. 4000 is a safe middle ground.
        
        Returns:
            A list of text chunks.
        """
        if not text:
            return []
        
        chunks = []
        
        # 1. Split by newlines first to get logical blocks (paragraphs)
        # Using re.split to keep the newlines for potential reconstruction if needed later,
        # but for chunking, we process the actual text between newlines.
        paragraph_splits = re.split(r'(\n+)', text)
        
        current_chunk_buffer = [] # To build up chunks of words
        current_chunk_length = 0

        for i, segment in enumerate(paragraph_splits):
            if not segment.strip(): # Skip empty segments or pure whitespace/newlines
                # If it's a newline segment, consider it as a separator for chunks
                if segment.strip() == '' and '\n' in segment:
                    if current_chunk_buffer: # If there's content in the buffer, finalize it
                        chunks.append(" ".join(current_chunk_buffer).strip())
                        current_chunk_buffer = []
                        current_chunk_length = 0
                    # Add the newline itself as a "chunk" if we want to preserve exact spacing,
                    # but typically we rejoin with " " or "\n\n" later.
                    # For actual translation chunks, we don't send just newlines.
                continue

            # Process actual text content
            words = segment.split(' ') # Simple word split, can be refined for other languages

            for word in words:
                word_len = len(word)
                # Check if adding the next word (plus a space) exceeds max_length
                # If current_chunk_buffer is empty, just add the word
                # Otherwise, add word + space
                
                # Check if the current word itself is too long for a chunk (very rare, e.g., chemical names)
                if word_len > max_length:
                    logger.warning(f"Very long word ({word_len} chars) encountered, splitting it arbitrarily.")
                    # Arbitrarily split the very long word
                    # This is a fallback for extreme cases, not ideal for linguistic sense
                    for j in range(0, word_len, max_length):
                        if word[j:j+max_length]:
                            chunks.append(word[j:j+max_length])
                    current_chunk_buffer = [] # Reset buffer after handling oversized word
                    current_chunk_length = 0
                    continue # Move to next word

                if current_chunk_buffer and (current_chunk_length + word_len + 1) > max_length: # +1 for space
                    # Current word won't fit, so finalize the current buffer
                    chunks.append(" ".join(current_chunk_buffer).strip())
                    current_chunk_buffer = [word]
                    current_chunk_length = word_len
                else:
                    # Add word to current buffer
                    current_chunk_buffer.append(word)
                    current_chunk_length += word_len + (1 if current_chunk_buffer else 0) # +1 for space if not first word

        # Add any remaining text in the buffer
        if current_chunk_buffer:
            chunks.append(" ".join(current_chunk_buffer).strip())
            
        # Filter out any empty chunks that might result from splitting logic
        chunks = [chunk for chunk in chunks if chunk]
        
        logger.info(f"Text split into {len(chunks)} chunks using newline and word-based strategy.")
        return chunks
        
    def translate_chunk(self, text: str, source_lang: str, retry_count: int = 3) -> str:
        """
        Translate a single chunk with retry logic
        """
        for attempt in range(retry_count):
            try:
                # Add delay for rate limiting
                if attempt > 0:
                    time.sleep(self.delay_between_requests * (attempt + 1))
                
                # Create translator instance for this chunk
                translator = GoogleTranslator(source=source_lang, target=self.target_lang)
                
                # Perform translation
                result = translator.translate(text)
                
                if result and result.strip():
                    return result.strip()
                else:
                    logger.warning(f"Empty result on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.warning(f"Translation attempt {attempt + 1} failed: {str(e)}")
                if attempt == retry_count - 1:
                    return f"[Translation failed: {str(e)}]"
                
                # Wait longer before retry
                time.sleep(self.delay_between_requests * 2)
                
        return "[Translation failed after all retries]"
    
    def translate_tafsir(self, 
                        input_text: str, 
                        source_language: Optional[str] = None,
                        preserve_structure: bool = True) -> Dict[str, Union[str, int, List, float]]:
        """
        Main method to translate tafsir text with automatic language detection
        
        Args:
            input_text: The text to translate
            source_language: Optional manual language specification (ar/ur)
            preserve_structure: Whether to preserve text structure
        """
        logger.info("Starting tafsir translation with automatic language detection...")
        
        # Detect language if not specified
        if source_language is None:
            detected_lang, confidence, lang_name = self.detect_language(input_text)
        else:
            detected_lang = source_language
            confidence = 1.0
            lang_name = self.supported_languages.get(source_language, source_language)
        
        logger.info(f"Source language: {lang_name} ({detected_lang})")
        
        # Preprocess text
        processed_text = self.preprocess_text(input_text, detected_lang)
        
        # Split into chunks
        chunks = self.split_text_intelligently(processed_text, detected_lang)
        logger.info(f"Text split into {len(chunks)} chunks")
        
        translated_chunks = []
        failed_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Translating chunk {i}/{len(chunks)}...")
            
            translated = self.translate_chunk(chunk, detected_lang)
            translated_chunks.append(translated)
            
            if translated.startswith("[Translation failed"):
                failed_chunks.append(i)
                logger.error(f"Failed to translate chunk {i}")
            
            # Small delay between chunks
            time.sleep(self.delay_between_requests)
        
        # Combine translated chunks
        if preserve_structure:
            full_translation = " ".join(translated_chunks)
        else:
            full_translation = "\n\n".join(translated_chunks)
        
        # Post-process translation
        full_translation = self._post_process_translation(full_translation)
        
        # Calculate success metrics
        successful_chunks = len(chunks) - len(failed_chunks)
        success_rate = (successful_chunks / len(chunks)) * 100 if chunks else 0
        
        result = {
            'original_text': input_text,
            'processed_text': processed_text,
            'translated_text': full_translation,
            'detected_language': detected_lang,
            'language_name': lang_name,
            'detection_confidence': confidence,
            'total_chunks': len(chunks),
            'successful_chunks': successful_chunks,
            'failed_chunks': failed_chunks,
            'success_rate': success_rate,
            'translation_timestamp': datetime.now().isoformat(),
            'chunks_detail': [
                {
                    'chunk_id': i,
                    'original': chunk,
                    'translated': trans,
                    'success': not trans.startswith("[Translation failed")
                }
                for i, (chunk, trans) in enumerate(zip(chunks, translated_chunks), 1)
            ]
        }
        
        logger.info(f"Translation completed! Success rate: {success_rate:.1f}%")
        return result
    
    def _post_process_translation(self, text: str) -> str:
        """Post-process translation for better readability"""
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix punctuation spacing
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Capitalize sentences
        sentences = re.split(r'([.!?]+)', text)
        processed_sentences = []
        
        for i, sentence in enumerate(sentences):
            if i % 2 == 0 and sentence.strip():
                sentence = sentence.strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                processed_sentences.append(sentence)
            else:
                processed_sentences.append(sentence)
        
        return ''.join(processed_sentences).strip()
    
    def translate_file(self, 
                           input_file: str, 
                           output_file: str = None,
                           source_language: Optional[str] = None,
                           output_format: str = 'txt') -> Dict:
        """
        Translate text from file with automatic language detection
        """
        try:
            # Read input file
            input_path = Path(input_file)
            with open(input_path, 'r', encoding='utf-8') as f:
                input_text = f.read()
            
            logger.info(f"Read {len(input_text)} characters from {input_file}")
            
            # Translate
            result = self.translate_tafsir(input_text, source_language)
            
            # Save output
            if output_file:
                output_path = Path(output_file)
                
                if output_format.lower() == 'json':
                    # Save as JSON
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                else:
                    # Save as formatted text
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write("=" * 70 + "\n")
                        f.write("MULTI-LANGUAGE TAFSIR TRANSLATION\n")
                        f.write("=" * 70 + "\n\n")
                        
                        f.write(f"Source Language: {result['language_name']} ({result['detected_language']})\n")
                        f.write(f"Detection Confidence: {result['detection_confidence']:.2f}\n")
                        f.write(f"Translation Date: {result['translation_timestamp']}\n")
                        f.write(f"Success Rate: {result['success_rate']:.1f}%\n")
                        f.write(f"Total Chunks: {result['total_chunks']}\n\n")
                        
                        f.write("ORIGINAL TEXT:\n")
                        f.write("-" * 40 + "\n")
                        f.write(result['original_text'])
                        f.write("\n\n")
                        
                        f.write("ENGLISH TRANSLATION:\n")
                        f.write("-" * 40 + "\n")
                        f.write(result['translated_text'])
                        
                        if result['failed_chunks']:
                            f.write(f"\n\nFAILED CHUNKS: {result['failed_chunks']}\n")
                
                logger.info(f"Translation saved to {output_file}")
            
            return result
            
        except Exception as e:
            logger.error(f"File translation error: {str(e)}")
            return {'error': str(e)}
    
    def translate_files(self, 
                             input_folder: str, 
                             output_folder: str = None,
                             file_pattern: str = "*.txt") -> Dict[str, Dict]:
        """
        Translate multiple files in a folder
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder) if output_folder else input_path / "translations"
        output_path.mkdir(exist_ok=True)
        
        results = {}
        
        for file_path in input_path.glob(file_pattern):
            logger.info(f"Processing file: {file_path.name}")
            
            output_file = output_path / f"translated_{file_path.stem}.txt"
            result = self.translate_file(str(file_path), str(output_file))
            
            results[file_path.name] = result
        
        # Save batch summary
        summary_file = output_path / "batch_translation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Batch translation completed. Summary saved to {summary_file}")
        return results

def main():
    """
    Example usage and demonstrations
    """
    # Sample texts in different languages
    arabic_sample = """
    قال الله تعالى: "وما أرسلناك إلا رحمة للعالمين". هذه الآية الكريمة تبين أن النبي محمد صلى الله عليه وسلم 
    قد أرسل رحمة للعالمين جميعاً، فرسالته عامة لجميع الناس في كل زمان ومكان. وهذا يدل على عظمة هذا الدين 
    وشموليته لجميع جوانب الحياة.
    """
    
    urdu_sample = """
    اللہ تعالیٰ نے فرمایا: "اور ہم نے تمہیں تمام جہانوں کے لیے رحمت بنا کر بھیجا ہے"۔ یہ آیت کریمہ اس بات کو 
    واضح کرتی ہے کہ حضرت محمد صلی اللہ علیہ وسلم کو تمام عالمین کے لیے رحمت بنا کر بھیجا گیا ہے۔ آپ کی 
    تعلیمات تمام انسانیت کے لیے ہیں اور ہر دور میں قابل عمل ہیں۔
    """
    
    print("=== MULTI-LANGUAGE TAFSIR TRANSLATOR ===\n")
    
    translator = TafsirTranslator()
    
    # Example 1: Arabic Translation with Auto-Detection
    print("1. Arabic Text Translation (Auto-Detection):")
    print("-" * 60)
    
    arabic_result = translator.translate_tafsir(arabic_sample)
    print(f"Detected Language: {arabic_result['language_name']} ({arabic_result['detected_language']})")
    print(f"Detection Confidence: {arabic_result['detection_confidence']:.2f}")
    print(f"Success Rate: {arabic_result['success_rate']:.1f}%")
    print("\nTranslation:")
    print(arabic_result['translated_text'])
    
    # Example 2: Urdu Translation with Auto-Detection
    print("\n" + "="*70)
    print("2. Urdu Text Translation (Auto-Detection):")
    print("-" * 60)
    
    urdu_result = translator.translate_tafsir(urdu_sample)
    print(f"Detected Language: {urdu_result['language_name']} ({urdu_result['detected_language']})")
    print(f"Detection Confidence: {urdu_result['detection_confidence']:.2f}")
    print(f"Success Rate: {urdu_result['success_rate']:.1f}%")
    print("\nTranslation:")
    print(urdu_result['translated_text'])
    
    # Example 3: Manual Language Specification
    print("\n" + "="*70)
    print("3. Manual Language Specification:")
    print("-" * 60)
    
    manual_result = translator.translate_tafsir(arabic_sample, source_language='ar')
    print(f"Manually set language: {manual_result['language_name']}")
    print("This is useful when auto-detection might be uncertain.")
    
    # Example 4: File Translation
    print("\n" + "="*70)
    print("4. File Translation Example:")
    print("-" * 60)
    
    # Save sample to file
    with open('sample_tafsir.txt', 'w', encoding='utf-8') as f:
        f.write(arabic_sample + "\n\n" + urdu_sample)
    
    file_result = translator.translate_file('sample_tafsir.txt', 'translated_output.txt')
    if 'error' not in file_result:
        print("✓ File translation completed!")
        print(f"✓ Detected: {file_result['language_name']}")
        print(f"✓ Success rate: {file_result['success_rate']:.1f}%")
        print("✓ Output saved to: translated_output.txt")
    
    print("\n" + "="*70)
    print("USAGE TIPS:")
    print("- The translator automatically detects Arabic or Urdu")
    print("- You can manually specify language with source_language='ar' or 'ur'")
    print("- Large texts are automatically chunked for better processing")
    print("- Failed chunks are logged and can be retried individually")

if __name__ == "__main__":
    print("REQUIRED PACKAGES:")
    print("pip install deep-translator")
    print("=" * 50)
    print()
    
    main()