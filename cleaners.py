import re

class EccoCleaner():

	def clean(self, text, info):
		if info == "NoClean":
			return text
		elif info == "Basic":
			return self.basic_clean(text)
		
	def basic_clean(self, text):
		text = re.sub("[^a-zA-Z ]", " ", text, flags=re.DOTALL)
		text = " ".join(text.split())
		return text