import re
import string
import cv2
import json
import os
import uuid

from pathlib import Path
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureNamedKeyCredential

from difflib import SequenceMatcher
from datetime import datetime
from dateutil import parser  
from loguru import logger

from ultralytics import YOLO
from openocr import OpenOCR  

from doc_verify.dataset import DOCUMENT_TYPES
from doc_verify.config import (
    TEMP_DIR,
    ACCOUNT, 
    ACCOUNT_KEY, 
    CONTAINER_NAME
)

class DocumentVerifier:
    
    def __init__(self, user_info: dict, cls_model_path: str):
        
        required_keys = [
            "document_type", 
            "image_path", 
            "name", 
            "surname", 
            "date_of_birth", 
            "gender"
        ]
        
        for key in required_keys:
            if key not in user_info:
                raise ValueError(f"Missing required key in user_info: {key}")
        
        self.document_type_id = user_info.get("document_type")
        
        if self.document_type_id not in DOCUMENT_TYPES:
            raise ValueError(f"Invalid document_type: {self.document_type_id}. Must be 1-6.")
        
        self.doc_type_name    = DOCUMENT_TYPES[self.document_type_id]
        self.image_path       = user_info.get("image_path")
        self.name             = user_info.get("name").upper()
        self.surname          = user_info.get("surname").upper()
        self.date_of_birth    = user_info.get("date_of_birth")
        self.expiration_date  = user_info.get("expiration_date")
        self.gender           = (user_info.get("gender") or "").upper()
        
        self.cls_model_path = cls_model_path

        try:
            models_dir_path = Path(cls_model_path).parent
            logger.info(f"Checking model weights in {models_dir_path} ...")
            os.makedirs(models_dir_path, exist_ok=True)
            
            if not any(file.endswith(".pt") for file in os.listdir(models_dir_path)):
                logger.info(f"Folder {models_dir_path} is empty. Start downloading weights from azure ...")
                
                self.download_blobs(
                    account=ACCOUNT,
                    account_key=ACCOUNT_KEY,
                    container_name=CONTAINER_NAME,
                    dest_dir=models_dir_path
                )
        except Exception as e:
            logger.error(f"Error downloading model weights from azure: {e}")
                
                
        try:
            self.model = YOLO(self.cls_model_path)
            logger.info(f"Classification model loaded from {self.cls_model_path}")
        except Exception as e:
            logger.error(f"Error loading classification model: {e}")
            raise e
        
        try:
            self.ocr_engine = OpenOCR(backend='onnx', device='cpu')
            logger.info("OCR engine initialized")
        except Exception as e:
            logger.error(f"Error initializing OCR: {e}")
            raise e
        
        self.formatted_date = self._format_expiration_date()
    
    
    def _format_expiration_date(self) -> datetime | None:
        """
        Formats expiration_date based on document type.
        
        Returns:
            datetime | None: Formatted date or None if not provided.
        """
        if not self.expiration_date:
            return None
        try:
            # document_type_1, document_type_3, document_type_5 use standard format
            if self.doc_type_name in ["document_type_1", "document_type_3", "document_type_5"]:
                return datetime.strptime(self.expiration_date, "%d/%m/%Y")
            # document_type_2 uses month/year format
            elif self.doc_type_name == "document_type_2":
                parts = self.expiration_date.split("/")
                if len(parts) >= 2:
                    month, year = parts[-2], parts[-1]
                    return datetime.strptime(f"01/{month}/{year}", "%d/%m/%Y")
                raise ValueError("Invalid expiration format for document_type_2")
            else:
                return None
        except ValueError as e:
            raise ValueError(f"Invalid expiration_date format: {e}")

    @staticmethod
    def download_blobs(
        account: str,
        account_key: str,
        container_name: str,
        dest_dir: Path
    ):
        
        cred = AzureNamedKeyCredential(
            account, 
            account_key
        )
        
        svc = BlobServiceClient(
            account_url=f"https://{account}.blob.core.windows.net",
            credential=cred
        )
        
        container = svc.get_container_client(container_name)
        
        for blob in container.list_blobs():
            
            blob_name = blob.name
            dest_path = dest_dir / blob.name
            
            with open(dest_path, "wb") as file:
                
                blob_client = container.get_blob_client(blob=blob_name)
                download_stream = blob_client.download_blob()
                
                file.write(download_stream.readall())

    @staticmethod
    def normalize_text(
        s: str
    ) -> str:
        '''
        Normalizes text: removes non-letter characters and converts to lowercase.
        
        Args:
            s (str): Input string.
        Returns:
            str: Normalized string.
        '''
        
        return re.sub(r'[^a-z]', '', s.lower())
    
    @staticmethod
    def fuzzy_contains(
        haystack:  str, 
        needle:    str, 
        threshold: float = 0.85
        ) -> bool:
        """
        Checks if a word exists in text with fuzzy matching.
        
        Args:
            haystack (str): Text to search in.
            needle (str): Word to search for.
            threshold (float): Matching threshold (0-1).
        Returns:
            bool: True if match found, False otherwise.
        """
        
        haystack_clean = re.sub(
            f"[{re.escape(string.punctuation)}]", 
            " ", 
            haystack.lower()
        )
        haystack_words = haystack_clean.split()
        needle_norm    = DocumentVerifier.normalize_text(needle)
        
        for word in haystack_words:
            if SequenceMatcher(
                None, 
                DocumentVerifier.normalize_text(word), 
                needle_norm
            ).ratio() >= threshold:
                return True
        return False
    
    @staticmethod
    def contains_sequence(
        haystack: str, 
        needle:   str
    ) -> bool:
        """
        Checks if a character sequence exists in text (ignoring spaces/punctuation).
        
        Args:
            haystack (str): Text to search in.
            needle (str): Sequence to search for.
        Returns:
            bool: True if sequence found, False otherwise.
        """
        
        haystack_clean = re.sub(
            r'[^A-Z]', 
            '', 
            haystack.upper()
        )
        
        needle_clean = re.sub(
            r'[^A-Z]', 
            '', needle.upper()
        )
        
        return needle_clean in haystack_clean
    
    @staticmethod
    def check_date_formats(
        date_obj: datetime, 
        text: str
    ) -> tuple[bool, list]:
        """
        Checks if a date exists in text in various formats.
        
        Args:
            date_obj (datetime): Date to check.
            text (str): Text to search in.
        Returns:
            tuple: (True, formats) if date found, (False, formats) otherwise.
        """
        
        clean_text = re.sub(r'[^A-Za-z0-9]', '', text.lower())
        formats = [
            "%Y/%m/%d", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y",
            "%B %d %Y", "%b %d %Y", "%B %d,%Y", "%b %d,%Y",
            "%d %B %Y", "%d %b %Y", "%d%B%Y", "%d%b%Y",
            "%B%d%Y", "%b%d%Y", "%B%d,%Y", "%b%d,%Y",
            "%d %B,%Y", "%d %b,%Y"
        ]
        
        for fmt in formats:
            try:
                candidate = date_obj.strftime(fmt).lower()
                candidate_clean = re.sub(
                    r'[^a-z0-9]', 
                    '', 
                    candidate
                )
                
                if candidate_clean in clean_text:
                    return True, formats
            except:
                continue
        return False, formats

    @staticmethod
    def extract_month_year(
        text: str, 
        keyword_mode: bool = False, 
        expiration_date: str | None = None
    ) -> bool:
        """
        Finds dates in text and compares with expiration_date in MM/YYYY format.
        
        Args:
            text (str): Text to search in.
            keyword_mode (bool): If True, search only near keywords (for document_type_2).
            expiration_date (str): Expected date in "MM/YYYY" format.
        Returns:
            bool: True if matching date found, False otherwise.
        """
        if not expiration_date:
            return False
    
        try:
            expected_date = datetime.strptime(expiration_date, "%m/%Y")
        except ValueError:
            return False
    
        candidates = []
        text_lower = text.lower()
    
        # Keyword mode — search only near keywords
        if keyword_mode:
            # Example: renewal on 02/15/2024
            renewal_match = re.findall(r"renewal\s*on\s*(\d{2})/(\d{2})/(\d{4})", text_lower)
            for m, d, y in renewal_match:
                try:
                    dt = datetime(int(y), int(m), 1)
                    candidates.append(dt)
                except ValueError:
                    continue
    
            # Example: renewal on 02/15/2024 to 02/15/2025
            renewal_range = re.findall(r"renewal\s*on.*?(\d{2})/(\d{2})/(\d{4}).*?(\d{2})/(\d{2})/(\d{4})", text_lower)
            for *_, m2, d2, y2 in renewal_range:
                try:
                    dt = datetime(int(y2), int(m2), 1)
                    candidates.append(dt)
                except ValueError:
                    continue
    
        # General patterns
        patterns = [
            r"(\d{2})/(\d{4})",  # MM/YYYY
            r"(\d{2})/(\d{2})/(\d{4})",  # MM/DD/YYYY
            r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[ .,/-]*(\d{1,2})[ ,/-]*(\d{4})",  # MAY 1 2026
        ]
    
        for pattern in patterns:
            for match in re.findall(pattern, text_lower):
                try:
                    if len(match) == 2:
                        m, y = map(int, match)
                        dt = datetime(int(y), int(m), 1)
                        candidates.append(dt)
    
                    elif len(match) == 3:
                        if isinstance(match[0], str) and match[0].isalpha():
                            month_str, day, year = match
                            dt = parser.parse(f"{month_str} {day} {year}")
                            candidates.append(datetime(dt.year, dt.month, 1))
                        else:
                            m, d, y = map(int, match)
                            dt = datetime(int(y), int(m), 1)
                            candidates.append(dt)
                except Exception:
                    continue
    
        # Compare each found date with expiration_date
        for dt in candidates:
            if dt.month == expected_date.month and dt.year == expected_date.year:
                return True
    
        return False

    @staticmethod
    def fuzzy_match_name(
        text: str, 
        name: str, 
        surname: str, 
        doc_type_specific: bool = False
    ) -> bool:
        """
        Checks name and surname in text using fuzzy matching.
        
        Args:
            text (str): Text to search in.
            name (str): Name to search for.
            surname (str): Surname to search for.
            doc_type_specific (bool): If True, allow combined name+surname.
        Returns:
            bool: True if name and surname found, False otherwise.
        """
        
        text_upper = text.upper()
        ok_name = DocumentVerifier.fuzzy_contains(text, name)
        ok_surname = DocumentVerifier.fuzzy_contains(text, surname)

        # MRZ check (e.g., P<PHLTAN<<FRANKLIN)
        if not (ok_name and ok_surname):
            mrz = [line for line in text_upper.split() if line.startswith("P<")]
            if mrz:
                mrz_line = mrz[0]
                if surname.upper() in mrz_line and name.upper() in mrz_line:
                    return True

        # For document_type_5: check combined name+surname
        if doc_type_specific and not (ok_name and ok_surname):
            combined = name + surname
            combined_clean = re.sub(r'[^A-Z]', '', combined.upper())
            haystack_clean = re.sub(r'[^A-Z]', '', text_upper)
            if combined_clean in haystack_clean:
                return True

        return ok_name and ok_surname
    
    @staticmethod
    def match_gender(
        text: str, 
        sex: str | None
    ) -> bool:
        """
        Gender may be absent — skip check (True).
        If present, looks for 'GENDER'/'SEX' keywords and expects values ('M'/'MALE' or 'F'/'FEMALE').
        
        Args:
            text (str): Text to search in.
            sex (str | None): Expected gender ('M', 'F' or None).
        Returns:
            bool: True if gender matches or not provided, False otherwise.
        """
        
        if not sex:  
            return True

        normalized_gender_map = {
            "M": ["M", "MALE"],
            "F": ["F", "FEMALE"]
        }
        expected_variants = normalized_gender_map.get(sex.upper(), [sex.upper()])

        possible_keywords = ["GENDER", "SEX"]
        lines = text.upper().splitlines()

        found_gender_keyword = any(keyword in line for line in lines for keyword in possible_keywords)
        if not found_gender_keyword:
            return True

        for i, line in enumerate(lines):
            for keyword in possible_keywords:
                if keyword in line:
                    for variant in expected_variants:
                        if variant in line:
                            return True
                    if i + 1 < len(lines):
                        if any(variant in lines[i + 1] for variant in expected_variants):
                            return True
        return False
    
    @staticmethod
    def match_dob(
        text: str, 
        date_of_birth: str
    ) -> bool:        
        """
        Checks if date of birth exists in text in various formats.
        
        Args:
            text (str): Text to search in.
            date_of_birth (str): Date of birth in "DD/MM/YYYY" format.
        Returns:
            bool: True if date found, False otherwise.
        """
        
        try:
            dob = datetime.strptime(date_of_birth, "%d/%m/%Y")
        except ValueError:
            return False
    
        text_clean = re.sub(r'[^A-Za-z0-9]', '', text.upper())
        text_words = re.findall(r'\w+', text.upper())
    
        formats = [
            "%Y/%m/%d", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y",
            "%B%d,%Y", "%b%d,%Y", "%B %d,%Y", "%b %d,%Y",
            "%d%B%Y", "%d%b%Y", "%d %B %Y", "%d %b %Y",
            "%B%d%Y", "%b%d%Y", "%B %d %Y", "%b %d %Y",
        ]
    
        for fmt in formats:
            try:
                dob_str = dob.strftime(fmt).upper()
                dob_str_clean = re.sub(r'[^A-Z0-9]', '', dob_str)
                if dob_str_clean in text_clean:
                    return True
                if dob_str in text.upper():
                    return True
            except:
                continue
    
        return False
    
    
    def verify_image(
        self, 
        threshold: float = 0.6
    ) -> bool:
        """
        Main document verification method.
        Pipeline:
            1. YOLO classification (with rotations if needed)
            2. If back side (document_type_4) — early exit (True)
            3. OCR + content verification (name, surname, DOB, gender, expiration)
            4. Rotations only for OCR (if verification fails)
        
        Args:
            threshold (float): Confidence threshold for classification.
        Returns:
            bool: True if document verified, False otherwise.
        """

        def predict_yolo_class(
            image_path: str
        ) -> tuple[bool, float, int]:
            try:
                results = self.model.predict(image_path, verbose=False)
                if results:
                    probs = results[0].probs
                    if probs:
                        predicted_label = int(probs.top1)
                        confidence = float(probs.top1conf.item())
                        return predicted_label == 1, confidence, predicted_label
                return False, 0.0, -1
            except Exception as e:
                return False, 0.0, -1

        def try_rotations_yolo(original_image, threshold=0.6):
            angles = [
                90, 
                -90, 
                180
            ]
            
            for ang in angles:
                rotated = cv2.rotate(
                    original_image, {
                        90: cv2.ROTATE_90_CLOCKWISE,
                        -90: cv2.ROTATE_90_COUNTERCLOCKWISE,
                        180: cv2.ROTATE_180
                    }[ang]
                )
                
                os.makedirs(TEMP_DIR, exist_ok=True)
                unique_filename = f"temp_rotated_yolo_{uuid.uuid4().hex}.jpg"
                temp_path = TEMP_DIR / unique_filename
                
                cv2.imwrite(temp_path, rotated)
                is_correct_class, confidence, predicted_label = predict_yolo_class(temp_path)
                
                if is_correct_class and confidence > threshold:
                    return temp_path
                else:
                    try:
                        os.remove(temp_path)
                    except FileNotFoundError:
                        pass
            
            return None

        def try_rotations_ocr(original_image) -> bool:
            angles = [
                90, 
                -90, 
                180
            ]
            
            for ang in angles:
                rotated = cv2.rotate(original_image, {
                    90: cv2.ROTATE_90_CLOCKWISE,
                    -90: cv2.ROTATE_90_COUNTERCLOCKWISE,
                    180: cv2.ROTATE_180
                }[ang])
                
                os.makedirs(TEMP_DIR, exist_ok=True)
                unique_filename = f"temp_rotated_ocr_{uuid.uuid4().hex}.jpg"
                temp_path = TEMP_DIR / unique_filename
                
                cv2.imwrite(temp_path, rotated)
                
                try:
                    text = get_text(temp_path)
                finally:
                    try:
                        os.remove(temp_path)
                    except FileNotFoundError:
                        pass
                
                if self._verify_document_content(text):
                    return True
            
            return False

        def get_text(path: str) -> str:
            try:
                                
                results, _ = self.ocr_engine(path)
                
                if isinstance(results, list):
                    if all(isinstance(r, dict) and "transcription" in r for r in results):
                        return " ".join(r["transcription"] for r in results).upper()
                    elif isinstance(results[0], str):
                        try:
                            raw = results[0].split("\t", 1)[-1].strip()
                            parsed = json.loads(raw)
                            return " ".join(r["transcription"] for r in parsed).upper()
                        except Exception:
                            return results[0].upper()
                elif isinstance(results, dict) and "transcription" in results:
                    return results["transcription"].upper()
                return str(results).upper()
            except Exception:
                return ""

        def get_doc_type_name() -> str:
            return DOCUMENT_TYPES.get(self.document_type_id, "UNKNOWN")

        def is_back_side() -> bool:
            return self.document_type_id == 4

        # === START PIPELINE ===
        processed_path = self.image_path
        temp_rotated_path = None
        doc_type = get_doc_type_name()
        formatted_date = self.formatted_date

        # Step 1: YOLO classification
        is_correct_class, confidence, predicted_label = predict_yolo_class(processed_path)

        if not is_correct_class or confidence <= threshold:
            img = cv2.imread(processed_path)
            rotated_path = try_rotations_yolo(img, threshold)
            if not rotated_path:
                logger.warning("YOLO failed even after rotations.")
                return False
            processed_path = rotated_path
            temp_rotated_path = rotated_path

        # Step 2: If back side (document_type_4) — early exit
        if is_back_side():
            logger.info("Document back side detected. Skipping OCR.")
            
            if temp_rotated_path:
                try:
                    os.remove(temp_rotated_path)
                except FileNotFoundError:
                    pass
            return True    
        

        # Step 3: OCR + verification
        text = get_text(processed_path)
        if self._verify_document_content(text):
            
            if temp_rotated_path:
                try:
                    os.remove(temp_rotated_path)
                except FileNotFoundError:
                    pass
            
            return True

        # Step 4: Rotations only for OCR
        img = cv2.imread(processed_path)
        result = try_rotations_ocr(img)
        
        if temp_rotated_path:
            try:
                os.remove(temp_rotated_path)
            except FileNotFoundError:
                pass
        
        return result


    def _verify_document_content(
        self, 
        text: str
    ) -> bool:
        """
        Verifies document content based on document type.
        Checks name, surname, date of birth, gender, expiration date (if applicable).
        
        Args:
            text (str): Text extracted by OCR.
        Returns:
            bool: True if all checks pass, False otherwise.
        """
        
    
        doc_type = self.doc_type_name
        text_upper = text.upper()

        name_ok = self.fuzzy_match_name(
            text, 
            self.name, 
            self.surname, 
            doc_type_specific=(doc_type in ["document_type_2", "document_type_3", "document_type_5"])
        )
        
        dob_ok = self.match_dob(text, 
            self.date_of_birth
        )
        
        gender_ok = self.match_gender(
            text, 
            self.gender
        )

        if not name_ok:
            name_ok = self.contains_sequence(text_upper, self.name) and self.contains_sequence(text_upper, self.surname)

        match doc_type:
            
            case "document_type_1":
                found_expiry, _ = self.check_date_formats(self.formatted_date, text) if self.formatted_date else (False, [])
                if self.formatted_date and not found_expiry:
                    try:
                        dob = datetime.strptime(self.date_of_birth, "%d/%m/%Y")
                        fallback_date = self.formatted_date.replace(day=dob.day, month=dob.month)
                        digits_only = re.sub(r'[^0-9]', '', text)
                        if str(fallback_date.year) in digits_only:
                            found_expiry = True
                    except Exception:
                        pass
                return name_ok and dob_ok and found_expiry

            case "document_type_2":
                same_month_year = self.extract_month_year(
                    text=text,
                    keyword_mode=True,
                    expiration_date=self.formatted_date.strftime("%m/%Y") if self.formatted_date else None
                )
                return name_ok and same_month_year

            case "document_type_3":
                found_expiry, _ = self.check_date_formats(self.formatted_date, text) if self.formatted_date else (False, [])
                return name_ok and dob_ok and gender_ok and found_expiry

            case "document_type_4":
                return name_ok

            case "document_type_5":
                phrase_clean = re.sub(
                    r'[^A-Z]', 
                    '', 
                    "Certificate of live birth".upper()
                )
                
                text_clean = re.sub(
                    r'[^A-Z]', 
                    '', 
                    text_upper
                )
                
                cert_found = phrase_clean in text_clean
                
                return cert_found or (name_ok and dob_ok and gender_ok)

            case "document_type_6":
                return self.contains_sequence(text_upper, self.name) and self.contains_sequence(text_upper, self.surname)

            case _:
                return False