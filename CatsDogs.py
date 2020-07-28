import numpy as np
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import preprocessing



class Interface():

    def Captura(self):
        self.filename = askopenfilename()
        self.image = Image.open(self.filename)
        self.photo = ImageTk.PhotoImage(self.image)
        label = Label(self.root, image=self.photo).grid(row=1, column=0, padx=15, pady=5, rowspan=3)
        label.image = self.photo


    def Treinamento(self):
        #LOCALIZAÇÃO DOS MÉTODOS
        self.classificador = Sequential()
        # DEFINIÇÃO DO KERNEL  E TAMANHO DA IMAGEM
        self.classificador.add(Conv2D(64, (3, 3), input_shape=(64, 64, 3), activation='relu'))

        # APLICAÇÃO DO MAXPOLLING
        self.classificador.add(BatchNormalization())
        self.classificador.add(MaxPooling2D(pool_size=(2, 2)))
        
        self.classificador.add(Flatten())

        #Criação de neurônios
        self.classificador.add(Dense(units=100, activation='relu'))
        self.classificador.add(Dense(units=100, activation='relu'))
        self.classificador.add(Dense(units=1, activation='sigmoid')) #Retorno de valores entre 0 e 1

        self.classificador.compile(optimizer='adam', loss='binary_crossentropy',
                              metrics=['accuracy'])

        #Gerador de imagens para treino
        gerador_treinamento = ImageDataGenerator(rescale=1. / 255,
                                                 rotation_range=7,
                                                 horizontal_flip=True,
                                                 shear_range=0.2,
                                                 height_shift_range=0.07,
                                                 zoom_range=0.2)

        gerador_teste = ImageDataGenerator(rescale=1. / 255)

        #Criação de bases de treinamento
        base_treinamento = gerador_treinamento.flow_from_directory('Database/Training',
                                                            target_size = (64,64),
                                                            batch_size = 32,
                                                            class_mode = 'binary')

        # Criação de bases de teste
        base_teste = gerador_treinamento.flow_from_directory('Databse/Teste',
                                                            target_size = (64,64),
                                                            batch_size = 32,
                                                            class_mode = 'binary')

        #Inicialização do treinamento
        self.classificador.fit_generator(base_treinamento, steps_per_epoch=20,
                                    epochs=3, validation_data=base_teste,
                                    validation_steps=20)
        #Contador de treino
        self.contador = 1


    def ClassificarImagens(self):

        imagem_teste = image.load_img(self.filename,
                                   target_size = (64, 64))
          
     
        imagem_teste = image.img_to_array(imagem_teste)
        imagem_teste /= 255

        #Criação de interface
        imagem_teste = np.expand_dims(imagem_teste, axis= 0)
        #Criação de predição
        previsao = self.classificador.predict(imagem_teste)
        
    
    
    
        if previsao > 0.5:
            print('A imagem é de um gato')
            Label(self.root, text='A imagem é de um gato').grid(row=1, column=0)
        elif previsao < 0.5:
            print('A imagem é de um cachorro')
            Label(self.root, text='A imagem é de um cachorro').grid(row=1, column=0)
            

    def __init__(self):
        self.contador = 0
        self.root = Tk()
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW")
        self.root.title("Identificaçao de imagens")
        


        Button(self.root,text='selecione a imagem', command=self.Captura).grid(row=0, column=0, pady=5)

        Button(self.root, text='Treinar Rede', command=self.Treinamento, width=10, height=2).grid(row=0, column=1)

        Button(self.root,text='Classificar', command=self.ClassificarImagens, width=10, height=2).grid(row=1, column=1)

        self.root.mainloop()



Interface()
