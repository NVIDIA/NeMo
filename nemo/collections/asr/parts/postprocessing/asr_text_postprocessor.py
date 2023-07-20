from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Type, Optional

class AbstractTextPostProcessor(ABC):
    language_id: str = None
    processor_registry: dict = {}

    @abstractmethod    
    def remove_punctutation(self, text):
        """
        Implemented by subclass in order to
        remove punctuation from text

        Args:
            text: input string

        Returns:
            A processed text string.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def remove_captialization(self, text):
        """
        Implemented by subclass in order to
        remove punctuation from text

        Args:
            text: input string

        Returns:
            A processed text string.
        """
        raise NotImplementedError()

    def normalize(self, text, is_remove_punctuation = False, is_remove_capitalization = False):

        if is_remove_punctuation:
            text = self.remove_punctutation(text)

        if is_remove_capitalization:
            text = self.remove_captialization(text)

        return text
    
    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)
    
    @classmethod
    def get_processor(cls, language_id: str) -> Optional[Type['AbstractTextPostProcessor']]:

        processor = None

        if language_id in cls.processor_registry:
            processor = cls.processor_registry[language_id]
        
        return processor
    
    @classmethod
    def register_language_processor(cls, text_postprocessor_cls: 'Type[AbstractTextPostProcessor]'):

        name = text_postprocessor_cls.__name__
        language_id = text_postprocessor_cls.language_id

        if language_id is None:
            raise ValueError(f"Cannot register `{name}` - language class variable is not set !")

        if language_id in cls.processor_registry:
            raise KeyError(f"Normalizer for `{language_id}` already exists in registry !")

        cls.processor_registry[language_id] = text_postprocessor_cls


    @classmethod
    def register_processor(cls):

        if cls.get_processor(cls.language_id) is None:
            cls.register_language_processor(cls)
        else:
            raise KeyError(f"Registration failed: Normalizer for `{cls.language_id}` already exists!")


    
@dataclass
class TextPostProcessorConfig:
    lang: str = "default"
    remove_punctuation: bool = False
    remove_capitalization: bool = False



