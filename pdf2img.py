from PIL import Image 
import pytesseract
from pdf2image import convert_from_path 




# PDF_file = "/home/shiva/Sirius/PDF/Division orders/19049.pdf"
# pages = convert_from_path(PDF_file, 500) 
# image_counter = 1 
# for page in pages: 
# 	filename = "page_"+str(image_counter)+".jpg" 
# 	page.save(filename, 'JPEG') 
# 	image_counter = image_counter + 1 
# filelimit = image_counter-1
# outfile = "/home/shiva/Sirius/Text/Division orders/19049.txt"
# f = open(outfile, "a") 
# for i in range(1, filelimit + 1): 
# 	filename = "page_"+str(i)+".jpg"
# 	text = str(((pytesseract.image_to_string(Image.open(filename))))) 
# 	text = text.replace('-\n', '')	 
# 	f.write(text) 
# f.close() 


import os,glob

def convertMultiple(pdfDir, txtDir):
    if pdfDir == "": pdfDir = os.getcwd() + "\\" 
    for pdf in os.listdir(pdfDir): 
        fileExtension = pdf.split(".")[-1]
        if fileExtension == "pdf":
            pdfFilename = pdfDir + pdf 
            text = convert_from_path(pdfFilename)
            image_counter = 1
            for page in text:
                filename = "page_"+str(image_counter) + ".jpg"
                page.save(filename,'JPEG')
                image_counter = image_counter + 1
            filelimit = image_counter - 1
            for i in range(1, filelimit +1):
                filename = "page_"+str(i)+".jpg"
                text = str(((pytesseract.image_to_string(Image.open(filename)))))
                text = text.replace('-\n','')
            textFilename = txtDir + pdf[-9:-4] + ".txt"
            textFile = open(textFilename, "w") #make text file
            textFile.write(str(text)) #write text to text file

pdfDir = "/home/shiva/Sirius/PDF/Division orders/"
txtDir = "/home/shiva/Sirius/Text/Division orders/"
str(convertMultiple(pdfDir, txtDir))