import tkinter as tk
from PIL import ImageTk, Image

from Crypto.Cipher import AES

import base64

root = tk.Tk()
root.title("Cryptography")


class Win1:

    def __init__(self, master):
        self.master = master
        self.master.geometry("580x640+10+10")
        self.img = ImageTk.PhotoImage(Image.open("eg.jpg").resize((580, 260)))
        self.l0 = tk.Label(self.master, text='Encryption using simple AES cipher', font=("times", 24, "bold"), bg='blue',
                      fg='white')
        self.l0.pack(side="top", fill="both", expand="yes")
        self.panel = tk.Label(self.master, image=self.img)
        self.panel.pack(side="top", fill="both", expand="yes")
        self.frame = tk.Frame(self.master)
        self.l1 = tk.Label(self.master,text='Protect your valuable data from hackers by encryption',font=("times",16,"bold"),fg='blue')
        self.l1.pack()
        self.butnew("Click to Encrypt",  Win2)
        self.butnew("Click to Decrypt",  Win3)
        self.frame.pack(expand="true")


    def butnew(self, text, _class):
        tk.Button(self.frame, text=text,command=lambda: self.new_window( _class),width=15,height=3,font=("times",14,"bold"), bg="purple",fg="yellow").pack(side="left",padx=15)

    def new_window(self,  _class):
        self.new = tk.Toplevel(self.master)
        _class(self.new)


class Win2:
    def encrypts(self):
        
        
        


        message = str(self.t1.get("1.0", tk.END))
        pass_phrase= self.tkey.get()
        secret_key=pass_phrase.ljust(16,'0').encode('utf-8')
        data=message.encode('utf8')
        
        
        cipher = AES.new(key, AES.MODE_CBC)
        ct_bytes = cipher.encrypt(pad(data, AES.block_size))
        ct = b64encode(ct_bytes).decode('utf-8')
        iv = b64encode(cipher.iv).decode('utf-8')
       

        self.t2.delete('1.0', tk.END)
        self.t2.insert("1.0", ct)
        self.t4.delete('1.0', tk.END)
        self.t4.insert("1.0", iv)

    def __init__(self, master):
        self.master = master
        self.master.geometry("550x500+470+150")
        self.master.title("Encrypt your data")
        self.master.configure(background="palegreen")
        self.frame = tk.Frame(self.master,borderwidth=2)
        self.l1 = tk.Label(self.master, text='Plain Text (Text to Encrypt)',font=("times", 16, "bold"),bg="palegreen", fg='red2')
        self.l1.pack(expand="yes")
        self.t1 = tk.Text(self.master, height=5, width=40, borderwidth=1,relief="solid")
        self.t1.pack(expand="yes")

        self.lkey = tk.Label(self.master, text='Secret key', font=("times", 16, "bold"), bg="palegreen", fg='red2')
        self.lkey.pack(expand="yes")
        self.tkey = tk.Entry(self.master, width=55,show="*", borderwidth=1, relief="solid")
        self.tkey.pack(expand="yes")

        self.b1 = tk.Button(self.master,text="Encrypt",command=self.encrypts,font=("times", 16, "bold"),bg="maroon",fg="white",width=10)
        self.b1.pack(expand="yes")
        self.l2 = tk.Label(self.master, text='Cipher Text (Encrypted Text)', font=("times", 16, "bold"),bg="palegreen", fg='red2')
        self.l2.pack(expand="yes")
        self.t2 = tk.Text(self.master, height=8, width=40, borderwidth=1, relief="solid",state="normal")
        self.t2.pack(expand="yes")
        
        self.l3 = tk.Label(self.master, text='IV  Initialization Vector', font=("times", 16, "bold"),bg="palegreen", fg='red2')
        self.l3.pack(expand="yes")
        self.t4 = tk.Text(self.master, height=8, width=40, borderwidth=1, relief="solid",state="normal")
        self.t4.pack(expand="yes")
        self.quit = tk.Button(self.frame, text="Close", command=self.close_window,font=("times", 16, "bold"),bg="maroon",fg="white",width=10)
        self.quit.pack()
        self.frame.pack(expand="true")

    def close_window(self):
        self.master.destroy()


class Win3:

    def decrypts(self):

        ciphertext = self.t1.get("1.0", tk.END)
        ct=b64decode(ciphertext)
        
        iv=self.t2.get("1.0",tk.END)
        iv=b64decode(iv)
        
       
        
        pass_phrase = self.tkey.get()
        secret_key=pass_phrase.ljust(16,'0').encode('utf-8')
        cipher = AES.new(secret_key, AES.MODE_CBC, iv)
        pt = unpad(cipher.decrypt(ct), AES.block_size)

        self.t3.delete('1.0', tk.END)
        self.t3.insert("1.0", pt)

    def __init__(self, master):
        self.master = master
        self.master.geometry("550x500+900+150")
        self.master.title("Decrypt your data")
        self.master.configure(background="pink")
        self.frame = tk.Frame(self.master, borderwidth=2)
        self.l1 = tk.Label(self.master, text='Cipher Text (Encrypted data)', font=("times", 16, "bold"), bg="pink",fg='blue')
        self.l1.pack(expand="yes")
        self.t1 = tk.Text(self.master, height=8, width=40, borderwidth=1, relief="solid")
        self.t1.pack(expand="yes")
        
        self.l2 = tk.Label(self.master, text='IV (Initialization Vector)', font=("times", 16, "bold"), bg="pink",fg='blue')
        self.l2.pack(expand="yes")
        self.t2 = tk.Text(self.master, height=8, width=40, borderwidth=1, relief="solid")
        self.t2.pack(expand="yes")

        self.lkey = tk.Label(self.master, text='Secret key', font=("times", 16, "bold"), bg="pink", fg='blue')
        self.lkey.pack(expand="yes")
        self.tkey = tk.Entry(self.master, width=55, show="*", borderwidth=1, relief="solid")
        self.tkey.pack(expand="yes")
        


        self.b1 = tk.Button(self.master, text="Decrypt", command=self.decrypts, font=("times", 16, "bold"), bg="blue",
                            fg="white", width=10)
        self.b1.pack(expand="yes")
        self.l3 = tk.Label(self.master, text='Plain Text(Original message)', font=("times", 16, "bold"), bg="pink",
                           fg='blue')
        self.l3.pack(expand="yes")
        self.t3 = tk.Text(self.master, height=5, width=40, borderwidth=1, relief="solid", state="normal")
        self.t3.pack(expand="yes")
        self.quit = tk.Button(self.frame, text="Close", command=self.close_window, font=("times", 16, "bold"),
                              bg="blue", fg="white", width=10)
        self.quit.pack()
        self.frame.pack(expand="true")

    def close_window(self):
        self.master.destroy()


app = Win1(root)
root.mainloop()