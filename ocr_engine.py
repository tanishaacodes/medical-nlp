from transformers import TrOCRProcessor,VisionEncoderDecoderModel
import torch

class MedReader:
    def __init__(self):
        self.processor=TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
        use_fast=False
        
        self.model=VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")

    def predict(self,pil_image):
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
            
        pixel_values=self.processor(images=pil_image,return_tensors="pt").pixel_values
        generated_ids=self.model.generate(pixel_values)
        generated_text=self.processor.batch_decode(generated_ids,skip_special_tokens=True)[0]
    # Inside your predict function:
        outputs = self.model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)

# This gets the probability of the generated tokens
        probs = torch.stack(outputs.scores, dim=1).softmax(-1)
        conf_score = torch.mean(torch.max(probs, dim=-1).values).item()
    
        return generated_text, conf_score
        
    
    
            

