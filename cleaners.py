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
	
class AssertioCleaner():
	
	def clean(self, text, info):
		if info == "LatinLib":
			return self.clean_latlib(text)
		elif info == "NewVer":
			return self.clean_new_ver(text)
		else:
			return self.clean_old_ver(text)
	
	def clean_latlib(self, text):
		for pattern_pair in [["\[(.*?)\]", " "], ["[,\.\?\!\(\)\\\]", " "], ["&", "et"]]:
			text = re.sub(pattern_pair[0], pattern_pair[1], text, flags=re.DOTALL)
		
		text_lines = []
		for line in text.split("\n"):
			#if line.isupper(): continue
			text_lines.append(line)
		text = " ".join(text_lines)
		
		
		words = []
		for word in text.split():
			if word.isalpha():
				words.append(word)

		
		text = " ".join(words)
		return text
	

	def clean_new_ver(self, text):
		lines = []
		for line in text.split("\n"):
			if line.startswith("To"): continue
			#elif line.isupper(): continue
			else: lines.append(line)
			
		text = " ".join(lines)
		
		blocks = re.findall("\{(.*?)\}", text)
		for block in blocks:
			if any(char.isdigit() for char in block):
				text = re.sub(block, " ", text, flags=re.DOTALL)
		
		text = re.sub("[,\.\!\?\(\)\"\{\};:‘'/\"\-\[\]]", " ", text, flags=re.DOTALL)
		text = re.sub("&", "et", text, flags=re.DOTALL)
		#print(text)	
			
		text_words = []
		for word in text.split():
			if word.isalpha():
				text_words.append(word)
		text = " ".join(text_words)
		
		return text
	
	def clean_old_ver(self, text):
		
		##Remove all special characters, replace & with et,
		for pattern_pair in [["[\{\}()\:\,\.\?\!\/]", ""], ["&", "et"]]:
			text = re.sub(pattern_pair[0], pattern_pair[1], text, flags=re.DOTALL)

		## Try to remove headers = line with all caps
		#text_lines = []
		#for line in text.split("\n"):
		#	if line.isupper(): continue
		#	else: text_lines.append(line)
		#text = " ".join(text_lines)

		## remove digits
		text_chars = []
		for char in text:
			if char.isalpha() or char == " ":
				text_chars.append(char)

		text = "".join(text_chars)
		text = " ".join(text.lower().split())

		return text
		