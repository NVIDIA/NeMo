from nemo.collections.asr.parts.postprocessing.asr_text_postprocessor import AbstractTextPostProcessor
import re

class GermanTextPostProcessor(AbstractTextPostProcessor):
    language_id = "de"

    def normalize_spaces(self, text):
        #regex for removing extra whitespaces:
        #"because  they're say-   we- at    least that's" -> "because they're say- we- at least that's"
        reg_remove_extra_space = '\s{2,}'
        text = re.sub(reg_remove_extra_space, ' ', text)
        
        #regex for removing whitespaces at the begining and at the end:
        #" because they're say- we- at least that's " -> "because they're say- we- at least that's"
        reg_remove_start_end_spaces = '\A\s|\s$'
        text = re.sub(reg_remove_start_end_spaces, '', text)
        
        return text
        
    def remove_punctutation(self, text):

        regex = r"[\.,\?]"
        text = self.normalize_spaces(re.sub(regex, ' ', text))
        return text
    
    def remove_captialization(self, text):

        text = text.lower()
        return text



# Registration step
GermanTextPostProcessor.register_processor()
