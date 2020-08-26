import tkinter as tk
import tkinter.filedialog as fd
import PIL.Image
import PIL.ImageTk
import sklearn.datasets
import sklearn.svm
import numpy

def imageToData(filename):
    greyImage = PIL.Image.open(filename).convert("L")
    greyImage = greyImage.resize((8,8),PIL.Image.ANTIALIAS)
    dispImage = PIL.ImageTk.PhotoImage(greyImage.resize((300,300)))
    imageLabel.configure(image = dispImage)
    imageLabel.image = dispImage
    numImage = numpy.asarray(greyImage, dtype = float)
    numImage = numpy.floor(16 - 16 * (numImage / 256))
    numImage = numImage.flatten()
    return numImage

def openFile():
    fpath = fd.askopenfilename()
    if fpath:
        data = imageToData(fpath)
        predictDigits(data)

def predictDigits(data):
    digits = sklearn.datasets.load_digits()
    clf = sklearn.svm.SVC(gamma = 0.001)
    clf.fit(digits.data, digits.target)
    n = clf.predict([data])
    textLabel.configure(text = "この画像は"+str(n)+"です！")

root = tk.Tk()
root.geometry("400x400")

btn = tk.Button(root,text="ファイルを開く",command = openFile)
imageLabel = tk.Label()

btn.pack()
imageLabel.pack()
textLabel = tk.Label(text="手書きの数字を認識します！")
textLabel.pack()

tk.mainloop()
