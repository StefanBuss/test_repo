import shutil
import pickle
import os
from PIL import Image, ImageDraw, ImageFont
import time 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import plotly.graph_objects as go
import scipy
from scipy import signal
from scipy.optimize import curve_fit
import sys
import random
import pandas as pd
import moviepy.video.io.ImageSequenceClip
import tensorflow as tf

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
import matplotlib.ticker as ticker

ROOT_DIR = os.getcwd()[:-8]
list_verwendete_daten = ["20221129_Unten_1_2_40","20221129_Unten_1_2_80"]

#für jeden Datensatz wird in einer liste definiert wie groß der volumenstrom der konti und dispersen phase ist und wie groß der hold-up ist 
#die liste für einen Datensatz ist folgendermaßen definiert: [volumenstrom_der_dispersen_phase [m^3/s], volumenstrom_der_kontinuierlichen_phase [m^3/s], hold_up]
list_konti_dispers_hold_up = {"20221129_Unten_1_2_40": [0.09/60000, 0.185/60000, 4.5 / 100], "20221129_Unten_1_2_80": [0.18/60000, 0.359/60000, 7.5 / 100]}


#ANPASSBARE PARAMETER
remove_data = True
konstante_framerate = False

step_size_gesch, untere_grenze_gesch, obere_grenze_gesch = 20, -100, 160

step_size, untere_grenze, obere_grenze = 0.2, 0, 3
untere_grenze_parametrisierung = 0.2 
obere_grenze_parametrisierung = 2.4

label_fontsize = 30
legend_fontsize = 20

if untere_grenze_parametrisierung < untere_grenze or obere_grenze_parametrisierung > obere_grenze:
    print("error - es muss gelten: untere_grenze_parametrisierung >= untere_grenze und obere_grenze_parametrisierung <= obere_grenze")
if obere_grenze_gesch < untere_grenze_gesch:
    print("error - es muss gelten: obere_grenze_gesch > untere_grenze_gesch")
if obere_grenze < untere_grenze:
    print("error - es muss gelten: obere_grenze > untere_grenze")


#parameter der definiert wie nah in pixeln ein Tropfen an den Rand vom Bildauschnit kommen darf. Falls er nicht in dem Bereich liegt wird er nicht mehr durch den Algorithmus berücksichtigt.
abstand_vom_rand = 10

#parameter die welche den Suchradius definieren
delta_seite = 0.9
delta_oben = 0.9
delta_unten = 0.9
maximale_abweichung_des_durchmessers = 0.3
maximale_abweichung_center = 0.7

#FIXE PARAMETER
alpha_gr = 8
d_um = 7.1 * 10 ** -3  # [m]
a15 = 1.52
a16 = 4.5
alpha_um = 10
#alpha_um = 8
D = 0.05 # [m] # column diameter
A_col = 0.25 * np.pi * D ** 2  # [m^2]

phi_st = 0.22
dh = 0.002
rho_c = 999 # [kg/m^3]
rho_d = 864.4 # [kg/m^3]
eta_d = 0.563*10**(-3)# [Pa*s]
eta_c = 0.939*10**(-3)  # [Pa*s]
sigma = 30.11* 10 ** -3  # [N/m]
g = 9.81            # [m/s^2]

pixelsize = 15.12162646 #[µm/px]
start = 0

#Hier wird Mask R-CNN genutzt um die Bounding Boxen aller angegeneben Datensätze zu berechnen.
def tropfen_erkennen(ROOT_DIR):
    print(ROOT_DIR)
    MODEL_DIR = ROOT_DIR + "\\droplet_logs"
    DATASET_INPUT_DIR = ROOT_DIR + "\\datasets\\input"
    DATASET_OUTPUT_DIR = ROOT_DIR + "\\datasets\\output"
    WEIGHTS_DIR = ROOT_DIR+"\\droplet_logs\\Netze_Tropfen\\mask_rcnn0_droplet_1305.h5"
    
    for VERWENDETE_DATEN in list_verwendete_daten:
            # is the evaluation done on cluster? 1 = yes, 0 = no
        cluster = 0
        masks = 0
        # max. image size
        # The multiple of 64 is needed to ensure smooth scaling of feature
        # maps up and down the 6 levels of the FPN pyramid (2**6=64),
        # e.g. 64, 128, 256, 512, 1024, 2048, ...
        # Select the closest value corresponding to the largest side of the image.
        image_max = 1024
        # is the evaluation done on CPU or GPU? 1=GPU, 0=CPU
        device = 1
        dataset_path = VERWENDETE_DATEN   
        weights_path = r"Netze_Tropfen"
        weights_name = r"mask_rcnn0_droplet_1305"

        tf.to_float = lambda x: tf.cast(x, tf.float32)
        # Root directory of the project
        if cluster == 0:
            IMAGE_MAX = image_max
            MASKS = masks
        else:
            import argparse
            # Parse command line arguments
            parser = argparse.ArgumentParser(
                    description='evaluation on cluster')
            parser.add_argument('--dataset_path', required=True,
                                    help='Dataset path to find in Mask_R_CNN\datasets\input')
            parser.add_argument('--save_path', required=True,
                                    help='Save path to find in Mask_R_CNN\datasets\output')
            parser.add_argument('--name_result_file', required=True,
                                    help='Name of the excel result file to find in Mask_R_CNN\datasets\output')
            parser.add_argument('--weights_path', required=True,
                                    help='Weights path to find in Mask_R_CNN\droplet_logs')
            parser.add_argument('--weights_name', required=True,
                                    help='Choose Neuronal Network / Epoch to find in Mask_R_CNN\droplet_logs')
            parser.add_argument('--masks', required=False,
                                default=0,
                                    help='Generate detection masks? 1 = yes, 0 = no')
            parser.add_argument('--device', required=False,
                                default=0,
                                help='is the evaluation done on CPU or GPU? 1=GPU, 0=CPU')
            parser.add_argument('--image_max', required=True,
                                default=1024,
                                help="max. image size")
            args = parser.parse_args()
            ROOT_DIR = os.path.join("/rwthfs/rz/cluster", os.path.abspath("../.."))
            WEIGHTS_DIR = os.path.join(ROOT_DIR, "droplet_logs", args.weights_path, args.weights_name + '.h5')
            DATASET_DIR = os.path.join(ROOT_DIR, "datasets/input", args.dataset_path)
            IMAGE_MAX = int(args.image_max)
            MASKS = int(args.masks)
                
        class DropletConfig(Config):
            """Configuration for training on the toy  dataset.
            Derives from the base Config class and overrides some values.
            """
            # Give the configuration a recognizable name
            NAME = "droplet"

            # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
            GPU_COUNT = 1

            # Generate detection masks
            #     False: Output only bounding boxes like in Faster-RCNN
            #     True: Generate masks as in Mask-RCNN
            
            if MASKS == 1:
                GENERATE_MASKS = True
            else: 
                GENERATE_MASKS = False
            
            # We use a GPU with 12GB memory, which can fit two images.
            # Adjust down if you use a smaller GPU.
            IMAGES_PER_GPU = 1

            # Number of classes (including background)
            NUM_CLASSES = 1 + 1  # Background + droplet

            # Skip detections with < 90% confidence
            DETECTION_MIN_CONFIDENCE = 0.7

            # Input image resizing
            IMAGE_MAX_DIM = IMAGE_MAX

        ### Configurations
        config = DropletConfig()
        config.display()

        ### Notebook Preferences

        # Device to load the neural network on.
        # Useful if you're training a model on the same 
        # machine, in which case use CPU and leave the
        # GPU for training.
        if device == 1:
            DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
        else:
            DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

        # Inspect the model in training or inference modes
        # values: 'inference' or 'training'
        # TODO: code for 'training' test mode not ready yet
        TEST_MODE = "inference"

        ### Load Model

        # Create model in inference mode
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                        config=config)
        # Load weights
        print("Loading weights ", WEIGHTS_DIR)
        model.load_weights(WEIGHTS_DIR, by_name=True)

        images = []                                                                              
        filenames = []
        filenames_sorted = os.listdir(DATASET_INPUT_DIR+"\\"+VERWENDETE_DATEN)
        filenames_sorted.sort() 
        for filename in filenames_sorted: 
            if not filename.endswith('.json'):
                image = cv2.imread(os.path.join(DATASET_INPUT_DIR+"\\"+VERWENDETE_DATEN, filename))
                images.append(image)                  
                filenames.append(filename)

        os.mkdir(DATASET_OUTPUT_DIR+"\\"+VERWENDETE_DATEN)
        for image_num, image in enumerate(images):                                              
            results = model.detect([image], verbose=1) 
            print(results)
            r = results[0]
            del r["masks"]
            pickle.dump(r, open(DATASET_OUTPUT_DIR+"\\"+VERWENDETE_DATEN+"\\"+filenames[image_num][6:-3], "wb"))

def erzeuge_titel_für_plots(VERWENDETE_DATEN):
    split_daten = VERWENDETE_DATEN.split("_")

    if split_daten[0] ==  "20221130":
        string_titel = "PV: " + split_daten[2] + "/" + split_daten[3]+ " - FP: " +  split_daten[4] + " - KP: " + split_daten[1] + " - V1"
    else:
        if split_daten[-2] == "ET":
            string_titel = "PV: " + split_daten[2] + "/" + split_daten[3]+ " - FP: " +  split_daten[4] + " - KP: " + split_daten[1] + " - ET: " + split_daten[-1]
        else:
            string_titel = "PV: " + split_daten[2] + "/" + split_daten[3]+ " - FP: " +  split_daten[4] + " - KP: " + split_daten[1]

    return string_titel
    
#Hier wird die unendliche Aufstiegsgeschwindigkeit für einen Durchmesser berechnet.
def get_v_infinity(diameter):       
    Ar = (rho_c * abs(rho_c - rho_d) * g * (diameter ** 3)) / (eta_c ** 2)                          
    c_w_unendlich = 432 / Ar + 20 / (Ar ** (1/3)) + (0.51 * (Ar ** (1/3))) / (140 + Ar ** (1/3))    
    Re_unendlich_blase = Ar / (12 * ((0.065 * Ar + 1) ** (1/6)))                                       
    Re_unendlich_kugel = np.sqrt(4/3 * Ar / c_w_unendlich)                                              
    f2 = 1 - (1 / (1 + (diameter / d_um) ** alpha_um))                                                  
    K_strich_Hr = (3 * (eta_c + eta_d / f2)) / (2 * eta_c + 3 * eta_d / f2)                            
    f1_strich = 2 * (K_strich_Hr - 1)                                                                 
    Re_unendlich_rund = (1 - f1_strich) * Re_unendlich_kugel + f1_strich * Re_unendlich_blase           
    
    v_gr = np.sqrt(abs(rho_c - rho_d) * g * diameter / (2 * rho_c))                                     
    v_os = np.sqrt(2 * a15 * sigma / (rho_c * diameter))                                                
    v_os_gr = (v_os ** alpha_gr + v_gr ** alpha_gr) ** (1 / alpha_gr)                                 
    v_rund = Re_unendlich_rund * eta_c / (rho_c * diameter)                                          

    v_infty = (v_os_gr * v_rund) / (v_os_gr ** a16 + v_rund ** a16) ** (1 / a16)                        

    return v_infty

#Hier wird der Verzögerungsfaktor für einne Durchmeser berechnet.
def get_k_v(diameter):
    pi_sigma = sigma * (rho_c ** 2 / (eta_c ** 4 * (rho_c - rho_d) * g)) ** (1 / 3)                                                 
    k_s = 1.406 * phi_st ** 0.145 * pi_sigma ** (-0.028) * np.exp(-0.129 * (diameter / dh) ** 1.134 * (1 - phi_st) ** (-2.161))     

    return k_s

#Hier wird die Aufstiegsgeschwindigkeit berechnet.
def get_u_y(diameter, n):
    u_x = V_x / (A_col * (1 - hold_up))
    k_s = get_k_v(diameter)
    v_infty = get_v_infinity(diameter)
    v_y = k_s * v_infty * (1 - hold_up) ** (n - 1) - u_x

    return v_y

#Hier werden die bounding boxen und namen der Bilder eines Datensatzes geladen.
#Außerdem wird das koordinatensystem angepasst. Zudem muss eine Boudning Box einen gewissen Randabstand einhalten um für die Tropfenverfolgung genutzt zu werden
def load_data():
    list_elem = []
    for elem in os.listdir(ROOT_DIR+"\\datasets\\output\\" + VERWENDETE_DATEN):
        list_elem.append(int(elem))
    
    list_elem.sort()

    size = Image.open(ROOT_DIR+"\\datasets\\input\\" + VERWENDETE_DATEN + "\\Image_" + str(list_elem[0])+".jpg").size
    bild_groeße_x, bild_groeße_y = int(size[0]), int(size[0])

    #HIER WERDEN NUR DIE BOUDNING BOXES SELEKTIERT, DIE NICHT ZU NAH AM RAND SIND
    list_r = []
    for elem in list_elem:
        r, r_selected = pickle.load(open(ROOT_DIR+"\\datasets\\output\\" + VERWENDETE_DATEN+"\\" + str(elem), "rb")), []
        
        for bounding_box in r["rois"]:
            height, width, center_y, center_x, durchmesser = morphologie_eines_tropfens(bounding_box)
            if (center_x - width/2 > abstand_vom_rand and center_x + width/2 < bild_groeße_x - abstand_vom_rand and
             center_y - height/2 > abstand_vom_rand and center_y + height/2 < bild_groeße_y - abstand_vom_rand):
                r_selected.append(list(bounding_box * pixelsize / 1000))
        list_r.append(np.array(r_selected))

    #HIER FINDET DIE UMDREHUNG DES KOORDINATENSYSTEMS STATT.
    r_selected_umgedrehtes_koordinantensystem = []
    for bild in list_r:
        bounding_boxen_mit_umgedrehter_y_achse = []
        for bounding_box in bild:
            bounding_boxen_mit_umgedrehter_y_achse.append([bild_groeße_y * pixelsize / 1000 - bounding_box[0], bounding_box[1], bild_groeße_y * pixelsize / 1000 - bounding_box[2], bounding_box[3]])
        
        r_selected_umgedrehtes_koordinantensystem.append(bounding_boxen_mit_umgedrehter_y_achse)
    #HIER FINDET DIE UMDREHUNG DES KOORDINATENSYSTEMS STATT.
        
    return r_selected_umgedrehtes_koordinantensystem, list_elem, len(list_elem) - 1

#Hier wird ein Plot erstellt, der die Unendliche Aufstiegeschwindigkeit in abhängigkeit vom durchmesser beschreibt. IST VERMUTLICH IRRELEVANT ZUM NACHZUVOLLZIEHEN
def plot_unendliche_aufstigsgeschwindigkeit():
    liste_durchmesser = np.linspace(0.05,9,100)
    liste_aufstiegsgeschwindigkeit = get_v_infinity(liste_durchmesser*10**(-3))

    fig, axs = plt.subplots(figsize=(16,9))
    axs.plot(liste_durchmesser, liste_aufstiegsgeschwindigkeit)

    axs.set_ylabel("Tropfenaufstiegsgeschwindigkeit [m/s]", fontsize = label_fontsize, fontname = "Arial")
    axs.set_xlabel("Tropfendurchmesser [mm]", fontsize = label_fontsize, fontname = "Arial")
    axs.set_title("Unendliche Tropfenaufstiegsgeschwindigkeit vom Stoffsystem Toluol, Wasser", fontsize = label_fontsize, fontname = "Arial")
    axs.tick_params(direction="in")

    for tick_label in axs.get_yticklabels():
        tick_label.set_fontsize(label_fontsize)
        tick_label.set_fontname('Arial')  

    for tick_label in axs.get_xticklabels():
        tick_label.set_fontsize(label_fontsize)
        tick_label.set_fontname('Arial')
        
    fig.tight_layout()
    plt.savefig(ROOT_DIR+ "\\datasets\\output_analyseddata\\Unendliche_Tropfenaufstiegsgeschwindigkeit", dpi = 300, bbox_inches='tight')

#hier wird ein Vergleich von Wiederholerversuchen durcgheführt. IST VERMUTLICH IRRELEVANT ZUM NACHZUVOLLZIEHEN
def plot_modell_wiederholder(y1,y2):
    bar_width = 0.35

    tropfengröße, x = np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2), []
    for i in range(len(tropfengröße)):
        if tropfengröße[i] != np.inf:
            string = str(np.round(tropfengröße[i] - step_size, 2)) + " bis " + str(tropfengröße[i])
        else:
            string = str(tropfengröße[i-1])+ " bis " + str(tropfengröße[i])
        x.append(string)
    
    fig, ax = plt.subplots(figsize=(16,9))
    ax.bar(np.arange(len(x)) - bar_width/2, y1, width=bar_width, label = "Relative Anzahl IDs je Größenklasse")
    ax.bar(np.arange(len(x)) + bar_width/2, y2, width=bar_width, label = "Relative Anzahl Tropfen je Größenklasse")

    ax.set_xlabel('Durchmesser [mm]', fontsize = label_fontsize, fontname = "Arial")
    ax.set_ylabel('Relative Anzahl', fontsize = label_fontsize, fontname = "Arial")
    ax.set_title(erzeuge_titel_für_plots(VERWENDETE_DATEN), fontsize = label_fontsize, fontname = "Arial")
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x)
    ax.legend(fontsize=legend_fontsize)
    ax.tick_params(direction="in",top=True, right=True, width=1.5, length=8)
    
    plt.xticks(rotation=45, ha='right', fontsize = label_fontsize, fontname = "Arial")
    plt.yticks(fontsize = label_fontsize, fontname = "Arial")
    fig.tight_layout()
    plt.subplots_adjust(bottom = 0.3)
    fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\vergleich_relative_anzahl_"+VERWENDETE_DATEN+".png", dpi = 300, bbox_inches='tight')
    plt.close()

#hier wird die relative anzahl von ids in einer größenklasse bezogen auf alle größenklassen berechnet (äußer der größenklasse unendlich)
def größenverteilung_id():
    dic_id_groeßenverteilung = pickle.load(open(ROOT_DIR+"\\datasets\\daten\\dic_id_groeßenverteilung_" + VERWENDETE_DATEN, "rb"))

    summe = 0
    for größe in dic_id_groeßenverteilung:
        if größe != np.inf:
            summe += len(dic_id_groeßenverteilung[größe])

    liste_relative_anzahl_id = []
    for größe in dic_id_groeßenverteilung:
        if größe != np.inf:
            liste_relative_anzahl_id.append(len(dic_id_groeßenverteilung[größe])/summe)

    return liste_relative_anzahl_id

#Hier wird die höhe, breite, mittelpunkt und durchmesser einer bounding box berechnet.
def morphologie_eines_tropfens(r):
    height = abs(r[2]-r[0])
    width = abs(r[3]-r[1])
    center_y = abs(r[0] + height/2)
    center_x = abs(r[1] + width/2)
    durchmesser = (height + width)/2

    return [height, width, center_y, center_x, durchmesser]

#Hier wird für jeden verfolgten Tropfen eines bildes die ID aus dic_id niedergeschrieben und bounding box dargestellt.
def show_image_draw_drop(bild_index, dic_id):
    input = ROOT_DIR+"\\datasets\\input\\" + VERWENDETE_DATEN + "\\Image_" + str(list_elem[bild_index])+".jpg"
    size = Image.open(input).size
    bild_groeße_x, bild_groeße_y = int(size[0]), int(size[0])
    
    im = Image.open(input).convert("RGBA")
    alle_tropfen_betrachtetes_bild = list_r[bild_index]
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(font = "arial.ttf", size = 24)

    for i in range(len(dic_id)):
        for ii in range(len(dic_id[i])):
            if dic_id[i][ii][0] == bild_index:
                tropfen_index = dic_id[i][ii][1]
                height, width, center_y, center_x, durchmesser = morphologie_eines_tropfens(alle_tropfen_betrachtetes_bild[tropfen_index])

                text = str(i)
                font_width, font_height = font.getsize(text)        #+ font_height/2
                #draw.text((center_x * 1000 / pixelsize - font_width/2,bild_groeße_y - center_y  * 1000 / pixelsize + font_height*2), text, fill = (255,0,0), font = font)
                draw.text((((alle_tropfen_betrachtetes_bild[tropfen_index][1] * 1000 / pixelsize)  + (alle_tropfen_betrachtetes_bild[tropfen_index][3] * 1000 / pixelsize))/2 - font_width/2 , - font_height/2 + ((bild_groeße_y - alle_tropfen_betrachtetes_bild[tropfen_index][2] * 1000 / pixelsize) + (bild_groeße_y - alle_tropfen_betrachtetes_bild[tropfen_index][0] * 1000 / pixelsize))/2), text, fill = (255,0,0), font = font)

                draw.rectangle((alle_tropfen_betrachtetes_bild[tropfen_index][1] * 1000 / pixelsize,
                                bild_groeße_y - alle_tropfen_betrachtetes_bild[tropfen_index][2] * 1000 / pixelsize,
                                alle_tropfen_betrachtetes_bild[tropfen_index][3] * 1000 / pixelsize,
                                bild_groeße_y - alle_tropfen_betrachtetes_bild[tropfen_index][0] * 1000 / pixelsize), outline = (255,0,0))
    
    im.save(ROOT_DIR+"\\datasets\\output_images\\" + VERWENDETE_DATEN + "\\Image_" + str(list_elem[bild_index]) + ".png")

#hier wird ein video erstellt, indem für jedes bild die korespondierenden ids der tropfenverfolgung niedergeschrieben wurden erstellt. IST VERMUTLICH IRRELEVANT ZUM NACHZUVOLLZIEHEN
def make_video(fps_make_video, image_folder, output_dir, save_dir):
    try:
        os.mkdir(output_dir)
    except:
        if remove_data == True:
            shutil.rmtree(output_dir)
            os.mkdir(output_dir)
        else:
            sys.exit("Datei gibt es schon")
        
    image_files = [os.path.join(image_folder,img) for img in os.listdir(image_folder) if img.endswith(".png")]
    image_files.sort()

    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps_make_video)
    clip.write_videofile(save_dir)

#hier wird ein video erstellt. IST VERMUTLICH IRRELEVANT ZUM NACHZUVOLLZIEHEN
def make_video_normalverteilung(fps_make_video, image_folder, output_dir):
    image_files = [os.path.join(image_folder,img) for img in os.listdir(image_folder) if img.endswith(".png") and img[-7:-4] in np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2).astype("str")]
    image_files.sort()

    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps_make_video)
    clip.write_videofile(output_dir)

#es wird das verhältnis aus Anzahl konsekutiv verfolgter Tropfen/Anzahl druch mrcnn detektierter Tropfen für jedes bild einens datensatzes berechnet und geplotet.
def verhaeltis_erkannte_bilder():
    list_all_erkannt = []
    list_all_tropfen = []
    verhaeltnis = []                #hier die funktion umschreiben. schneller laufen lassen
    for bild_index in range(start, ende):
        sum_tropfen_erkannt = 0
        for i in range(len(dic_id)):
            for ii in range(len(dic_id[i])):
                if dic_id[i][ii][0] == bild_index:
                    sum_tropfen_erkannt += 1
        
        list_all_tropfen.append(len(list_r[bild_index]))
        list_all_erkannt.append(sum_tropfen_erkannt)
        if sum_tropfen_erkannt != 0:
            verhaeltnis.append(sum_tropfen_erkannt/len(list_r[bild_index]))
        else:
            verhaeltnis.append(0)

    plot_funktion(zeit[:-1], [verhaeltnis], "Verfolgungsrate", "Zeit [s]", ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\verhaeltis_" + VERWENDETE_DATEN, 0, 1, erzeuge_titel_für_plots(VERWENDETE_DATEN))    

#hier wird die bedingung für eine potentielle "vorgänger-nachfolger" beziehung definiert. RELEVANT ZUM NACHVOLLZIEHEN
def Bedingung_vorgaenger_nachfolger(morph_aktuelles_bild, morph_naechstest_bild, alle_tropfen_betrachtetes_bild, alle_tropfen_naechstest_bild, tropfen_index_betrachtetes_bild, tropfen_index_naechstest_bild):
    height_current, width_current, center_y_current, center_x_current, durchmesser_current = morph_aktuelles_bild
    height_next, width_next, center_y_next, center_x_next, durchmesser_next = morph_naechstest_bild
    
    vergleich_druchmesser = durchmesser_current

    return (abs((durchmesser_next-durchmesser_current)/durchmesser_current) < maximale_abweichung_des_durchmessers and
                    
    center_y_current + vergleich_druchmesser * maximale_abweichung_center > center_y_next and
    center_y_current - vergleich_druchmesser * maximale_abweichung_center < center_y_next and
    center_x_current + vergleich_druchmesser * maximale_abweichung_center >  center_x_next and
    center_x_current - vergleich_druchmesser * maximale_abweichung_center < center_x_next)

#hier findet die konsekuitive tropfenverfolgung statt. Es ist das dictionary dic_id erstellt, indem die wahrschenlichsten vorgänger nachfolger beziehungen gespeichert werden.
#SEHR RELEVANT ZUM NACHVOLLZIEHEN
def konsekutive_bilderverfolgung():
    dic_id, id_counter = {}, 0
    try:
        os.mkdir(ROOT_DIR + "\\datasets\\output_images\\" + VERWENDETE_DATEN)
    except:
        if remove_data == True:
            shutil.rmtree(ROOT_DIR + "\\datasets\\output_images\\" + VERWENDETE_DATEN)
            os.mkdir(ROOT_DIR + "\\datasets\\output_images\\" + VERWENDETE_DATEN)
        else:
            sys.exit("Datei gibt es schon")
        
    list_komb = []
    for bild_index in range(start, ende):
        alle_potentiellen_vorgänger_nachfolger_beziehungen = []
        
        alle_tropfen_betrachtetes_bild = list_r[bild_index]
        alle_tropfen_naechstest_bild = list_r[bild_index + 1]

        #hier werden alle potentiellen vorgänger nachfolger beziehungen bestimmt 
        for tropfen_index_betrachtetes_bild in range(len(alle_tropfen_betrachtetes_bild)):
            alle_potentiellen_vorgänger_nachfolger_beziehungen.append([[bild_index, tropfen_index_betrachtetes_bild]])
            
        for tropfen_index_betrachtetes_bild in range(len(alle_tropfen_betrachtetes_bild)):
            morph_aktuelles_bild = morphologie_eines_tropfens(alle_tropfen_betrachtetes_bild[tropfen_index_betrachtetes_bild])
            
            for tropfen_index_naechstest_bild in range(len(alle_tropfen_naechstest_bild)):
                morph_naechstest_bild = morphologie_eines_tropfens(alle_tropfen_naechstest_bild[tropfen_index_naechstest_bild])
                
                if Bedingung_vorgaenger_nachfolger(morph_aktuelles_bild, morph_naechstest_bild, alle_tropfen_betrachtetes_bild, alle_tropfen_naechstest_bild, tropfen_index_betrachtetes_bild, tropfen_index_naechstest_bild) == True:
                    for i, value in enumerate(alle_potentiellen_vorgänger_nachfolger_beziehungen):
                        if value[0][0] == bild_index and value[0][1] == tropfen_index_betrachtetes_bild:
                            alle_potentiellen_vorgänger_nachfolger_beziehungen[i].append([bild_index + 1, tropfen_index_naechstest_bild])

        #Hier wird sortiert zwischen Potentiellen Nachfolger Beziehungen bei denen ein Vorgänger mehrere potentielle Nachfolger Bezeihungen hat und den übrigen potentiellen
        #vorgänger nachfolger beziehungen.
        #Es werden zudem alle Vorgänger rausgefiltert die keinen Nachfolger zugeordnet worden sind.
        alle_vorgänger_nachfolger_beziehungen_mit_mehreren_potentiellen_nachfolgern = [elem for elem in alle_potentiellen_vorgänger_nachfolger_beziehungen if len(elem) > 2]
        alle_potentiellen_vorgänger_nachfolger_beziehungen_außer_ein_vorgänger_mehrere_potentielle_nachfolger = [elem for elem in alle_potentiellen_vorgänger_nachfolger_beziehungen if len(elem) == 2]

        #Hier werden die wahrscheinlichsten Nachfolgern berechnet für alle potentielle Vorgänger Nachfolger Beziehungen bei denen ein Vorgänger mehrere potentielle Nachfolger hat 
        list_tasaelicher_nachfolger = []
        for pair in alle_vorgänger_nachfolger_beziehungen_mit_mehreren_potentiellen_nachfolgern:
            vorgaenger = pair[0]
            list_abweichung = []
            morph_aktuelles_bild = morphologie_eines_tropfens(alle_tropfen_betrachtetes_bild[vorgaenger[1]])
                      
            for ii in range(1,len(pair)):
                potentieller_nachfolger = pair[ii]
                morph_naechstest_bild = morphologie_eines_tropfens(alle_tropfen_naechstest_bild[potentieller_nachfolger[1]])
                abweichung = ((morph_naechstest_bild[3] - morph_aktuelles_bild[3])**2 + (morph_naechstest_bild[2] - morph_aktuelles_bild[2])**2) ** 0,5
                list_abweichung.append(abweichung)
            index_min = np.argmin(list_abweichung)
            list_tasaelicher_nachfolger.append([vorgaenger, pair[index_min + 1]])

        wahrscheinlichste_VNB_wenn_vorgänger_mehrere_nachfolger_und_alle_potentiellen_VNB_wenn_nachfolger_mehrere_potentielle_vorgänger_hat = alle_potentiellen_vorgänger_nachfolger_beziehungen_außer_ein_vorgänger_mehrere_potentielle_nachfolger + list_tasaelicher_nachfolger
        alle_verfolgten_nachfolger = []
        for pair in wahrscheinlichste_VNB_wenn_vorgänger_mehrere_nachfolger_und_alle_potentiellen_VNB_wenn_nachfolger_mehrere_potentielle_vorgänger_hat:
            if pair[1] not in alle_verfolgten_nachfolger:
                alle_verfolgten_nachfolger.append(pair[1])

        #hier wird der wahrschenlichste vorgänger berechnet für alle potentiellen vorgänger nachfolger beziehungen bei denen ein Nachfolger mehrere potentielle vorgänger hat.
        #Da so beeinhalte nach diesem schritt die liste "wahrscheinlichsten_vorgänger_nachfolger_beziehungen" eine eindeutige vorgänger nachfolger beziehung. 
        wahrscheinlichsten_vorgänger_nachfolger_beziehungen = []
        for nachfolger in alle_verfolgten_nachfolger:
            List_potentieller_vorgaenger = []
            for pair in wahrscheinlichste_VNB_wenn_vorgänger_mehrere_nachfolger_und_alle_potentiellen_VNB_wenn_nachfolger_mehrere_potentielle_vorgänger_hat:
                if nachfolger == pair[-1]:
                    List_potentieller_vorgaenger.append(pair[0])
            abweichung_center_potentieller_vorgaenger = []
            for elem in List_potentieller_vorgaenger:
                morph_aktuelles_bild = morphologie_eines_tropfens(alle_tropfen_betrachtetes_bild[tropfen_index_betrachtetes_bild])
                morph_naechstest_bild = morphologie_eines_tropfens(alle_tropfen_naechstest_bild[tropfen_index_naechstest_bild])
                abweichung = ((morph_naechstest_bild[3] - morph_aktuelles_bild[3])**2 + (morph_naechstest_bild[2] - morph_aktuelles_bild[2])**2) ** 0,5
                abweichung_center_potentieller_vorgaenger.append(abweichung)
            index_min = np.argmin(abweichung_center_potentieller_vorgaenger)
            vorgaenger = List_potentieller_vorgaenger[index_min]
            wahrscheinlichsten_vorgänger_nachfolger_beziehungen.append([vorgaenger, nachfolger])

        #speicherung der wahrschenlichsten vorgänger nachfolger beziehung über mehrere konsekuitive bilder 
        if dic_id == {}:
            for i, val in enumerate(wahrscheinlichsten_vorgänger_nachfolger_beziehungen):
                dic_id[id_counter] = val
                id_counter += 1
        else:
            for i, val in enumerate(wahrscheinlichsten_vorgänger_nachfolger_beziehungen):
                found = False 
                for elem in dic_id: 
                    if dic_id[elem][-1] == val[0]:
                        dic_id[elem].append(val[1])
                        found = True
                if found == False:
                    dic_id[id_counter] = val
                    id_counter += 1        

        show_image_draw_drop(bild_index, dic_id)
        list_komb.append(wahrscheinlichsten_vorgänger_nachfolger_beziehungen)

    pickle.dump(list_komb, open(ROOT_DIR+"\\datasets\\daten\\vorgaenger_nachfolger_beziehung_"+VERWENDETE_DATEN, "wb"))
    pickle.dump(dic_id, open(ROOT_DIR+"\\datasets\\daten\\dic_id_"+VERWENDETE_DATEN, "wb"))
    make_video(10, ROOT_DIR + "\\datasets\\output_images\\"+VERWENDETE_DATEN, ROOT_DIR + "\\datasets\\output_videos\\" + VERWENDETE_DATEN, ROOT_DIR + "\\datasets\\output_videos\\" + VERWENDETE_DATEN +'\\'+VERWENDETE_DATEN+".mp4") 

    return dic_id

#hier wird die geschwindigkeit von Tropfen über zwei konsekuitive Bilder mittels vorwärtsdifferenzen berechnet.  RELEVANT ZUM NACHVOLLZIEHEN
def geschwindigkeit_bild_index_bis_bild_index_plus_eins():
    list_v_x, list_v_y = [], []
    vorgaenger_nachfolger_beziehung = pickle.load(open(ROOT_DIR+"\\datasets\\daten\\vorgaenger_nachfolger_beziehung_"+VERWENDETE_DATEN, "rb"))

    for bild_index in range(start, ende):
        alle_tropfen_betrachtetes_bild = list_r[bild_index]
        alle_tropfen_naechstest_bild = list_r[bild_index + 1]
        
        sum_v_x, sum_v_y = 0, 0
        for pair in vorgaenger_nachfolger_beziehung[bild_index]:
            
            height_current, width_current, center_y_current, center_x_current, durchmesser_current = morphologie_eines_tropfens(alle_tropfen_betrachtetes_bild[pair[0][1]])
            height_next, width_next, center_y_next, center_x_next, durchmesser_next = morphologie_eines_tropfens(alle_tropfen_naechstest_bild[pair[1][1]])

            v_x = (center_x_next - center_x_current)/(zeit[bild_index + 1] - zeit[bild_index])
            v_y = (center_y_next - center_y_current)/(zeit[bild_index + 1] - zeit[bild_index])          

            sum_v_x += v_x
            sum_v_y += v_y

        if len(vorgaenger_nachfolger_beziehung[bild_index]) != 0:
            avg_v_x = sum_v_x/len(vorgaenger_nachfolger_beziehung[bild_index])          #ist dies zeile Falsch. einmal überprüfen. gibt es immer nur zweier paare. und zb keine einer paare. stimmt die länge oder muss durch len +1 geteilt werden?
            avg_v_y = sum_v_y/len(vorgaenger_nachfolger_beziehung[bild_index])
        else:
            avg_v_x = 0
            avg_v_y = 0
            
        list_v_x.append(avg_v_x)
        list_v_y.append(avg_v_y)

    return list_v_x, list_v_y

#hier wird die geschwindigkeit einer id berechnet also die durschnittliche geschwindigkeit über mehrere konsekuitive bilder.  RELEVANT ZUM NACHVOLLZIEHEN
def geschwindigkeit_einer_id(id):
    erster_bild_index = dic_id[id][0][0]
    erster_tropfen_index = dic_id[id][0][1]
    alle_tropfen_betrachtetes_bild = list_r[erster_bild_index]
    height_current, width_current, center_y_current, center_x_current, durchmesser_current = morphologie_eines_tropfens(alle_tropfen_betrachtetes_bild[erster_tropfen_index])

    letzter_bild_index = dic_id[id][-1][0]
    letzter_tropfen_index = dic_id[id][-1][1]
    alle_tropfen_letztes_bild = list_r[letzter_bild_index]
    height_last, width_last, center_y_last, center_x_last, durchmesser_last = morphologie_eines_tropfens(alle_tropfen_letztes_bild[letzter_tropfen_index])

    v_y = (center_y_last - center_y_current)/(zeit[letzter_bild_index] - zeit[erster_bild_index])
    v_x = (center_x_last - center_x_current)/(zeit[letzter_bild_index] - zeit[erster_bild_index])

    return v_y, v_x

#hier wird der durschnitliche sauterdurchmesser eines bildes berechnet.
def berechnen_sauterdurchmesser_pro_bild(bild_index):
    alle_tropfen_betrachtetes_bild = list_r[bild_index]

    if len(alle_tropfen_betrachtetes_bild) != 0:
        sum_d_hoch_drei, sum_d_hoch_zwei = 0, 0

        for tropfen_index, val in enumerate(alle_tropfen_betrachtetes_bild):
            height, width, center_y, center_x, durchmesser = morphologie_eines_tropfens(alle_tropfen_betrachtetes_bild[tropfen_index])
            sum_d_hoch_drei += durchmesser**3
            sum_d_hoch_zwei += durchmesser**2            

        return sum_d_hoch_drei/sum_d_hoch_zwei
    
    if len(alle_tropfen_betrachtetes_bild) == 0:
        return 0

#hier wird der sationäre sauterdurchemesser berechnet.
def stationaerer_sauterdurchmesser():
    sum_all_sauterdurchmesser = 0
    list_avg_sauterdurchmesser = []
    
    for bild_index in range(start, ende):
        
        aktueller_sauterdurchmesser = berechnen_sauterdurchmesser_pro_bild(bild_index)
        sum_all_sauterdurchmesser += aktueller_sauterdurchmesser 
        avg_sauterdurchmesser = sum_all_sauterdurchmesser / (bild_index + 1)

        list_avg_sauterdurchmesser.append(avg_sauterdurchmesser)

    plot_funktion(zeit[:-1], [list_avg_sauterdurchmesser], "Sauterdurchmesser [mm]", "Zeit [s]", ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\sauterdurchmesser_"+VERWENDETE_DATEN, None, None, erzeuge_titel_für_plots(VERWENDETE_DATEN))
    pickle.dump(list_avg_sauterdurchmesser[-1], open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\sauterdurchmesser", "wb"))

#hier wird der durschnittliche durchmesser einer id berechnet.  RELEVANT ZUM NACHVOLLZIEHEN
def avg_durchmesser_id(ID):
    sum_durchmesser = 0

    for i in range(len(dic_id[ID])):
        bild_index = dic_id[ID][i][0]
        tropfen_index = dic_id[ID][i][1]

        alle_tropfen_betrachtetes_bild = list_r[bild_index]
        height, width, center_y, center_x, durchmesser = morphologie_eines_tropfens(alle_tropfen_betrachtetes_bild[tropfen_index])
        sum_durchmesser += durchmesser
        #print("durchmesser: ", durchmesser)        Krasser unterschied zwischen durchmesser und avg_durchmesser

    avg_durchmesser = sum_durchmesser/(i+1)
    #print("avg_durchmesser: ", avg_durchmesser)        Krasser unterschied zwischen durchmesser und avg_durchmesser

    return avg_durchmesser

#hier werden die ids aus dic_id in größenklassen sortiert in dem dictionary dic_id_groeßenverteilung gespeichert. RELEVANT ZUM NACHVOLLZIEHEN
def nach_groeße_sortieren():
    dic_id_groeßenverteilung = {}

    for groeße in np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2):
        dic_id_groeßenverteilung[groeße] = []

    dic_id_groeßenverteilung[np.inf] = []

    for größe in np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2):
        for id in dic_id:
            avg_durchmesser = avg_durchmesser_id(id)
                
            if avg_durchmesser < größe and avg_durchmesser > größe - step_size:
                dic_id_groeßenverteilung[größe].append(id)

    for id in dic_id:
        avg_durchmesser = avg_durchmesser_id(id)
        if avg_durchmesser > obere_grenze:
            dic_id_groeßenverteilung[np.inf].append(id)

    pickle.dump(dic_id_groeßenverteilung, open(ROOT_DIR+"\\datasets\\daten\\dic_id_groeßenverteilung_"+VERWENDETE_DATEN, "wb"))
    return dic_id_groeßenverteilung

#hier wird die anzahl an ids je geschwindigkeitsklasse einer größenklasse dargestellt.
def AnzahlderTropfen_Geschwindigkeit_RGB():
    os.mkdir(ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\AnzahlderTropfen_Geschwindigkeit_RGB")

    dic_id_geschwindigkeitsverteilung_ges = {}
    
    dic_id_geschwindigkeitsverteilung_alle_größen = {}
    dic_id_geschwindigkeitsverteilung_alle_größen[-np.inf] = []
    for geschwindigkeit in np.round(np.arange(untere_grenze_gesch + step_size_gesch, obere_grenze_gesch + step_size_gesch, step_size_gesch), 2):
        dic_id_geschwindigkeitsverteilung_alle_größen[geschwindigkeit] = []
    dic_id_geschwindigkeitsverteilung_alle_größen[np.inf] = []

    max_anzahl = 0
    for größe in dic_id_groeßenverteilung:
        
        dic_id_geschwindigkeitsverteilung = {}
        dic_id_geschwindigkeitsverteilung[-np.inf] = []
        for geschwindigkeit in np.round(np.arange(untere_grenze_gesch + step_size_gesch, obere_grenze_gesch + step_size_gesch, step_size_gesch), 2):
            dic_id_geschwindigkeitsverteilung[geschwindigkeit] = []
        dic_id_geschwindigkeitsverteilung[np.inf] = []

        for geschwindigkeit in np.round(np.arange(untere_grenze_gesch + step_size_gesch, obere_grenze_gesch + step_size_gesch, step_size_gesch), 2):
            for ID in dic_id_groeßenverteilung[größe]:
                avg_gesch_y, avg_gesch_x  = geschwindigkeit_einer_id(ID)
                        
                if avg_gesch_y < geschwindigkeit and avg_gesch_y > geschwindigkeit - step_size_gesch:
                    dic_id_geschwindigkeitsverteilung[geschwindigkeit].append(ID)
                    dic_id_geschwindigkeitsverteilung_alle_größen[geschwindigkeit].append(ID)
                    
        for ID in dic_id_groeßenverteilung[größe]:
            avg_gesch_y, avg_gesch_x  = geschwindigkeit_einer_id(ID)
            if avg_gesch_y > obere_grenze_gesch:
                dic_id_geschwindigkeitsverteilung[np.inf].append(ID)
                dic_id_geschwindigkeitsverteilung_alle_größen[np.inf].append(ID)
            if avg_gesch_y < untere_grenze_gesch:
                dic_id_geschwindigkeitsverteilung[-np.inf].append(ID)
                dic_id_geschwindigkeitsverteilung_alle_größen[-np.inf].append(ID)
    
        dic_id_geschwindigkeitsverteilung_ges[größe] = dic_id_geschwindigkeitsverteilung

        zwischenspeicher = len(dic_id_geschwindigkeitsverteilung[max(dic_id_geschwindigkeitsverteilung, key=lambda k: len(dic_id_geschwindigkeitsverteilung[k]))])    
        summe = 0
        for key in list(dic_id_geschwindigkeitsverteilung.keys()):
            if key not in ["-inf", "inf"]:
                summe += len(dic_id_geschwindigkeitsverteilung[key])
                
        if max_anzahl < zwischenspeicher/summe:
            max_anzahl = zwischenspeicher/summe

    pickle.dump(dic_id_geschwindigkeitsverteilung_ges, open(ROOT_DIR+"\\datasets\\daten\\dic_id_geschwindigkeitsverteilung_ges_"+VERWENDETE_DATEN, "wb"))
    pickle.dump(dic_id_geschwindigkeitsverteilung_alle_größen, open(ROOT_DIR+"\\datasets\\daten\\dic_id_geschwindigkeitsverteilung_alle_größen_"+VERWENDETE_DATEN, "wb"))
    
    for größe in dic_id_geschwindigkeitsverteilung_ges:
        plot_geschwindigkeitsverteilung(dic_id_geschwindigkeitsverteilung_ges[größe], größe, max_anzahl)
    plot_geschwindigkeitsverteilung(dic_id_geschwindigkeitsverteilung_alle_größen, None, None)

#hier wird der rgb wert aller bilder eines datensatzes berechnet. 
def get_rgb():
    images = []                                                                              
    filenames = []
    filenames_sorted = os.listdir(ROOT_DIR + "\\datasets\\input\\" + VERWENDETE_DATEN)[start:ende]
    filenames_sorted.sort() 
    for filename in filenames_sorted: 
        image = cv2.imread(os.path.join(ROOT_DIR + "\\datasets\\input\\" + VERWENDETE_DATEN, filename))
        images.append(image)                  
        filenames.append(filename)

    rgb_list = []
    for image_num, image in enumerate(images):                                              
        rgb = image.mean(axis=0).mean(axis=0)[1] 
        rgb_list.append(rgb) 

    return rgb_list

#hier wird m_dis erstellt. erste spalte beeinhaltet größenklasse. zweite spalte beeinhaltet anzahl ids in der jeweiligen größenklasse. dritte spalte beeinhaltet die y aufstiegsgeschwindigkeit. vierte spalte beeinhaltet die x geschwindigkeit.
def size_distribution():
    step_list = np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2)
    M_dis = np.zeros(shape = (len(step_list) + 1, 6))
    M_dis[:-1,0] = step_list
    M_dis[len(step_list),0] = np.inf
    
    for i, größe in enumerate(dic_id_groeßenverteilung):
        M_dis[i,1] = len(dic_id_groeßenverteilung[größe])

        sum_v_y, sum_v_x = 0, 0
        sum_v_y_only_positive, sum_v_y_only_negative = 0, 0
        counter_only_positive, counter_only_negative = 0, 0
        
        for ID in dic_id_groeßenverteilung[größe]:
            v_y, v_x = geschwindigkeit_einer_id(ID)
            
            sum_v_y +=  v_y 
            sum_v_x +=  v_x
            if v_y > 0:
                sum_v_y_only_positive += v_y
                counter_only_positive += 1
            else:
                sum_v_y_only_negative += v_y
                counter_only_negative += 1

        if len(dic_id_groeßenverteilung[größe]) != 0:
            M_dis[i,2] = sum_v_y/len(dic_id_groeßenverteilung[größe])
            M_dis[i,3] = sum_v_x/len(dic_id_groeßenverteilung[größe])
        else:
            M_dis[i,2] = 0
            M_dis[i,3] = 0
            
        if counter_only_positive != 0:
            M_dis[i,4] = sum_v_y_only_positive/counter_only_positive
        else:
            M_dis[i,4] = 0

        if counter_only_negative != 0:
            M_dis[i,5] = sum_v_y_only_negative/counter_only_negative
        else:
            M_dis[i,5] = 0

    pickle.dump(M_dis, open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\M_dis", "wb"))

#eine funktionen mit denen graphen erstellt werden. VERMUTLICH IRRELEVANT ZUM NACHVOLZIEHEN. diese funktion könnte zu verwirrung führen, da unsauber geschrieben. intention schnell wenig code aber unübersichtlich
def plot_funktion(x_achse, y_achse, y_label, x_label, SAVE_DIR, unteres_y_lim, oberes_y_lim, title):
    fig, axs = plt.subplots(figsize=(16,9))
    
    for elem in y_achse:   
        axs.plot(x_achse, elem)

    if y_label == "Zeit in [s]":
        axs.plot(x_achse, y_achse[0], label = "tatsächliches Zeitverhalten", color = "blue")
        axs.plot(x_achse, y_achse[1], label = "Zeitverhalten bei 300 FPS", color = "orange")
        lines, labels = axs.get_legend_handles_labels()
        axs.legend(lines, labels , loc = 'upper left', fontsize=legend_fontsize)
 
    axs.set_ylabel(y_label, fontsize = label_fontsize, fontname = "Arial")
    axs.set_xlabel(x_label, fontsize = label_fontsize, fontname = "Arial")
    axs.set_title(title, fontsize = label_fontsize, fontname = "Arial")
    axs.tick_params(direction="in",top=True, right=True, width=1.5, length=8)    
    
    for tick_label in axs.get_yticklabels():
        tick_label.set_fontsize(label_fontsize)
        tick_label.set_fontname('Arial')  

    for tick_label in axs.get_xticklabels():
        tick_label.set_fontsize(label_fontsize)
        tick_label.set_fontname('Arial')
    
    if unteres_y_lim != None and oberes_y_lim != None:
        axs.set_ylim([unteres_y_lim, oberes_y_lim])

    fig.tight_layout()
    plt.savefig(SAVE_DIR, dpi = 300, bbox_inches='tight')
    plt.close()

#plot_funktion(np.linspace(0, len(zeit), len(zeit)), [zeit, np.linspace(0, 10, len(zeit)) ], "Zeit in [s]", "Bild_Index", ROOT_DIR + "\\datasets\\timestamps\\zeit_" + VERWENDETE_DATEN + ".png", None, None, erzeuge_titel_für_plots(VERWENDETE_DATEN))
        

#eine funktionen mit denen graphen erstellt werden
#VERMUTLICH IRRELEVANT ZUM NACHVOLZIEHEN, da sie zur schriftliche ausarbeitung der BA benötigt wird.
#diese funktion könnte zu verwirrung führen, da unsauber geschrieben. intention schnell wenig code aber unübersichtlich
def plot_bar(loc_label,label, pos_string,strings, x_achse, y_achse, y_modell, y_label, x_label, title, SAVE_DIR):
    fig, ax = plt.subplots(figsize=(16,9))
    if label != None:
        ax.bar(x_achse, y_achse, width = 0.8, label = label)
    else:
        ax.bar(x_achse, y_achse, width = 0.8)
        
    if y_modell != None:

        if y_modell[3] != None:
            ax.plot(x_achse, y_modell[0], label = y_modell[2],color='r')        
            ax.plot(x_achse, y_modell[1], label = y_modell[3],color='k')
        else:
            if "Parametriserung dieses Datensatzes" in y_modell[2]:
                ax.plot(x_achse, y_modell[0], label = y_modell[2],color='r')        
            if "Gesamt-Parametrisierung" in y_modell[2]:
                ax.plot(x_achse, y_modell[0], label = y_modell[2],color='k')
            
    ax.set_xlabel(x_label, fontsize = label_fontsize, fontname = "Arial")
    ax.set_ylabel(y_label, fontsize = label_fontsize, fontname = "Arial")
    ax.set_title(title, fontsize = label_fontsize, fontname = "Arial")
    ax.tick_params(direction="in",top=True, right=True, width=1.5, length=8)
    
    for i, string in enumerate(strings):
        ax.text(pos_string[0], pos_string[1] - i * 0.05, string, ha='left', va='top', transform=plt.gca().transAxes, fontsize = text_fontsize)

    lines, labels = ax.get_legend_handles_labels()
    
    if label != None:
        ax.legend(lines, labels , loc = loc_label, fontsize=legend_fontsize)  #'upper left'
       
    plt.yticks(fontsize = label_fontsize, fontname = "Arial")
    fig.tight_layout()
    plt.subplots_adjust(bottom = 0.3)


    umbenannte_liste_verwendete_daten = []
    for elem in list_verwendete_daten:
        umbenannte_liste_verwendete_daten.append(erzeuge_titel_für_plots(elem))
    
    if all(element in umbenannte_liste_verwendete_daten for element in x_achse):
        plt.xticks(rotation=45, ha='right', fontsize = 14, fontname = "Arial")
    else:
        plt.xticks(rotation=45, ha='right', fontsize = label_fontsize, fontname = "Arial")

    fig.savefig(SAVE_DIR, dpi = 300, bbox_inches='tight')
    plt.close()

#eine funktion die geschwindigkeit je größenklasse plottet und das literaturmodell mit plottet.
#VERMUTLICH IRRELEVANT ZUM NACHVOLZIEHEN, da sie zur schriftliche ausarbeitung der BA benötigt wird.
#diese funktion könnte zu verwirrung führen, da unsauber geschrieben. intention schnell wenig code aber unübersichtlich
def plot_size_distribution(): 
    M_dis = pickle.load(open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\M_dis", "rb"))
    step_list = list(np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2))
    tropfengröße = list(M_dis[:,0])
    erste_index = tropfengröße.index(step_list[0])
    letzter_index = tropfengröße.index(step_list[-1])

    anzahl_tropfen, v_y, x = list(M_dis[:,1])[erste_index:letzter_index+1], list(M_dis[:,2])[erste_index:letzter_index+1], []
    for i, größe in enumerate(step_list):
        x.append(str(np.round(größe - step_size, 2)) + " bis " + str(größe))

    plot_bar(loc_label = None, label = None,pos_string = [0.65,0.93], strings = [], x_achse = x, y_achse = anzahl_tropfen, y_modell =  None, y_label = 'Anzahl ID', x_label = 'Tropfengröße [mm]', title = erzeuge_titel_für_plots(VERWENDETE_DATEN) + " summe anzahl ID: " + str(sum(anzahl_tropfen)),SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\v_y_positive_negative_all_anzahl_tropfen\\anzahl_tropfen_"+VERWENDETE_DATEN)
    
    step_list = list(np.round(np.arange(untere_grenze_parametrisierung + step_size, obere_grenze_parametrisierung + step_size, step_size), 2))
    erste_index = tropfengröße.index(step_list[0])
    letzter_index = tropfengröße.index(step_list[-1])

    if VERWENDETE_DATEN.split("_")[4] != "100":
        y_modell_n, y_modell_n_ges, y_inf = [], [], []
        for diameter in step_list:
            y_modell_n.append(get_u_y(diameter*10**-3, n)*10**3)
            y_modell_n_ges.append(get_u_y(diameter*10**-3, n_ges)*10**3)
            y_inf.append(get_v_infinity(diameter*10**-3)*10**3)
    
    anzahl_tropfen, v_y, x = list(M_dis[:,1])[erste_index:letzter_index+1], list(M_dis[:,2])[erste_index:letzter_index+1], []
    for i, größe in enumerate(step_list):
        x.append(str(np.round(größe - step_size, 2)) + " bis " + str(größe))

    if VERWENDETE_DATEN.split("_")[4] != "100": 
        plot_bar(loc_label = "upper left", label = "gemessene Aufstiegsgeschwindigkeit", pos_string = [], strings = [],x_achse = x, y_achse = v_y, y_modell = [y_modell_n, y_modell_n_ges, "Parametriserung dieses Datensatzes, n = " + str(np.round(n,2)), "Gesamt-Parametrisierung, n = " + str(np.round(n_ges,2))], y_label = 'Tropfenaufstiegsgeschwindigkeit [mm/s]', x_label = 'Tropfengröße [mm]',title = erzeuge_titel_für_plots(VERWENDETE_DATEN), SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\v_y_positive_negative_all_anzahl_tropfen\\v_y_parametrisiert_"+VERWENDETE_DATEN)
        
        plot_bar(loc_label = "upper left", label = "gemessene Aufstiegsgeschwindigkeit", pos_string = [], strings = [],x_achse = x, y_achse = v_y, y_modell = [y_modell_n, y_inf,  "Parametriserung dieses Datensatzes, n = " + str(np.round(n,2)), "v_inf"], y_label = 'Tropfenaufstiegsgeschwindigkeit [mm/s]', x_label = 'Tropfengröße [mm]',title = erzeuge_titel_für_plots(VERWENDETE_DATEN), SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\v_y_positive_negative_all_anzahl_tropfen\\v_y_parametrisiert_v_inf_"+VERWENDETE_DATEN)

        plot_bar(loc_label = "upper left", label = "gemessene Aufstiegsgeschwindigkeit", pos_string = [], strings = [],x_achse = x, y_achse = v_y, y_modell = [y_modell_n, None,  "Parametriserung dieses Datensatzes, n = " + str(np.round(n,2)), None], y_label = 'Tropfenaufstiegsgeschwindigkeit [mm/s]', x_label = 'Tropfengröße [mm]',title = erzeuge_titel_für_plots(VERWENDETE_DATEN), SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\v_y_positive_negative_all_anzahl_tropfen\\v_y_parametrisiert_nur_modell_"+VERWENDETE_DATEN)
        plot_bar(loc_label = "upper left", label = "gemessene Aufstiegsgeschwindigkeit", pos_string = [], strings = [],x_achse = x, y_achse = v_y, y_modell = [y_modell_n_ges, None,  "Gesamt-Parametrisierung, n = " + str(np.round(n_ges,2)), None], y_label = 'Tropfenaufstiegsgeschwindigkeit [mm/s]', x_label = 'Tropfengröße [mm]',title = erzeuge_titel_für_plots(VERWENDETE_DATEN), SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\v_y_positive_negative_all_anzahl_tropfen\\v_y_parametrisiert_nur_gesamt_modell_"+VERWENDETE_DATEN)

    plot_bar(loc_label = "upper left", label = None, pos_string = [], strings = [],x_achse = x, y_achse = v_y, y_modell = None, y_label = 'Tropfenaufstiegsgeschwindigkeit [mm/s]', x_label = 'Tropfengröße [mm]',title = erzeuge_titel_für_plots(VERWENDETE_DATEN), SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\v_y_positive_negative_all_anzahl_tropfen\\v_y"+VERWENDETE_DATEN)

#plotten von anzahl an ids je geschwindigkeitsklasse einer größenklasse
#VERMUTLICH IRRELEVANT ZUM NACHVOLZIEHEN, da sie zur schriftliche ausarbeitung der BA benötigt wird.
#diese funktion könnte zu verwirrung führen, da unsauber geschrieben. intention schnell wenig code aber unübersichtlich
def plot_geschwindigkeitsverteilung(dict, größe, max_anzahl):
    x, keys = [], list(dict)
    for i in range(len(dict)):
        if keys[i] == np.inf:
            string = str(keys[i-1])+ " bis " + str(keys[i])
        elif keys[i] == -np.inf:
            string =  str(keys[i]) + " bis " + str(np.round(keys[i+1] - step_size_gesch, 2))
        else:
            string = str(np.round(keys[i] - step_size_gesch, 2)) + " bis " + str(keys[i])
        x.append(string)

    y, gesamt_anzahl_tropfen = [], 0
    for geschwindigkeit in dict:
        anzahl = len(dict[geschwindigkeit])
        y.append(anzahl)
        gesamt_anzahl_tropfen += anzahl

    if max_anzahl == None:
        max_anzahl = 0
        for elem in np.array(y)/gesamt_anzahl_tropfen:
            if elem > max_anzahl:
                max_anzahl = elem

    if größe == np.inf:
        p_text = "Größe: " + str(obere_grenze) + " bis " + str(größe) + " mm"
    elif größe == None:
        p_text = "Größe: alle größen"
    else:
        p_text = "Größe: " + str(np.round(größe - step_size, 2)) + " bis " + str(größe) + " mm"
    
    fig, ax = plt.subplots(figsize=(16,9))
    ax.bar(x, np.array(y)/gesamt_anzahl_tropfen, width = 0.8)
    ax.set_xlabel('Geschwindigkeit [mm/s]', fontsize = 24, fontname = "Arial")
    ax.set_ylabel('Relative Anzahl ID', fontsize = 24, fontname = "Arial")
    ax.set_title(erzeuge_titel_für_plots(VERWENDETE_DATEN) + ", " + p_text + ", " + "Summe ID: " + str(gesamt_anzahl_tropfen), fontsize = 24, fontname = "Arial")
    plt.xticks(rotation=45, ha='right', fontsize = 24, fontname = "Arial")
    plt.subplots_adjust(bottom = 0.16)
    ax.set_ylim([0, max_anzahl])
    ax.tick_params(direction="in",top=True, right=True, width=1.5, length=8)
    decimal_places = 2  # Anzahl der gewünschten Nachkommastellen
    formatter = ticker.FormatStrFormatter(f'%.{decimal_places}f')
    ax.yaxis.set_major_formatter(formatter)
    
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontsize(24)
        tick_label.set_fontname('Arial')

    fig.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\AnzahlderTropfen_Geschwindigkeit_RGB" + "\\AnzahlderTropfen_Geschwindigkeit_" + VERWENDETE_DATEN +"_"+ str(größe)+".png", dpi = 300, bbox_inches='tight')
    plt.close()

#hier wird ein sinus fit von daten mit einer Frequenz von 1,25 hz berechnet. Sie wird zum fitten von rgb und v_y benötigt. 
def fit_funktion(data, zeit):
    freq = 1.25
    
    guess_amplitude = 3*np.std(data)/(2**0.5)
    guess_phase = 0
    guess_offset = np.mean(data)

    p0 = [guess_amplitude, guess_phase, guess_offset]

    def my_sin(x, amplitude, phase, offset):
        return np.sin(x * 2 * np.pi * freq + phase) * amplitude + offset

    fit = curve_fit(my_sin, zeit, data, p0=p0)
    data_fit = my_sin(np.linspace(0, zeit[-1], len(zeit)), *fit[0])
    
    return data_fit, fit[0][1]

#hier wird die berechnete geschwindigkeit mittels vorwärtsdifferenzen zwischen zwei bilder für einen Datensatz und der rgb wert geplottet. die variable bereich gibt an wie viele sekunden des datenstatzes geplottet werden soll. Es nur ein gekürtzer bereich betrachtet, da sonst zu ünübersichtlich
def plot_rgb_v_y_gekuertzt(list_v_y, zeit):
    bereich = 4
    for i, val in enumerate(zeit):
        if val > bereich:
            break
        
    fittet_v_y, phase_v_y = fit_funktion(list_v_y[:i], zeit[:i])
    fittet_rgb, phase_rgb = fit_funktion(rgb_list[:i], zeit[:i])
    fig, ax1 = plt.subplots(figsize=(16,9))
    
    ax1.plot(zeit[:i], rgb_list[:i], 'b',linestyle='none', marker = "o", label = "RGB-Farbwert")
    ax1.plot(np.linspace(0, zeit[i], len(zeit[:i])), fittet_rgb, 'b-', label = "Sinus Fit von RGB-Farbwert")
    ax1.set_xlabel('Zeit [s]', fontsize = label_fontsize, fontname = "Arial")
    ax1.set_ylabel('RGB-Farbwert', color='b', fontsize = label_fontsize, fontname = "Arial")
    ax1.tick_params('y', colors='b',direction="in")
    ax1.tick_params(direction="in",top=True, right=True, width=1.5, length=8)
    
    for tick_label in ax1.get_yticklabels():
        tick_label.set_fontsize(label_fontsize)
        tick_label.set_fontname('Arial')  

    for tick_label in ax1.get_xticklabels():
        tick_label.set_fontsize(label_fontsize)
        tick_label.set_fontname('Arial')
    
    ax2 = ax1.twinx()
    ax2.plot(zeit[:i], list_v_y[:i], 'k',linestyle='none', marker = "o", label = "Aufstiegsgeschwindigkeit")
    ax2.plot(np.linspace(0, zeit[i], len(zeit[:i])), fittet_v_y, 'k-', label = "Sinus Fit der Aufstiegsgeschwindigkeit")
    
    ax2.set_ylabel('Tropfenaufstiegsgeschwindigkeit [mm/s]', color='k', fontsize = label_fontsize, fontname = "Arial")
    ax2.set_ylim([-40, 80])
    ax2.tick_params('y', colors='k',direction="in")

    for tick_label in ax2.get_yticklabels():
        tick_label.set_fontsize(label_fontsize)
        tick_label.set_fontname('Arial')  

    for tick_label in ax2.get_xticklabels():
        tick_label.set_fontsize(label_fontsize)
        tick_label.set_fontname('Arial')
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.set_title(erzeuge_titel_für_plots(VERWENDETE_DATEN), fontsize = label_fontsize, fontname = "Arial")
    ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=legend_fontsize)
    ax2.tick_params(direction="in",top=True, right=True, width=1.5, length=8)
    
    fig.tight_layout()

    plt.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\rgb_v_y_gekuertzt_"+VERWENDETE_DATEN, dpi = 300)
    plt.close()

    fig, ax2 = plt.subplots(figsize=(16,9))
    ax2.plot(zeit[:i], list_v_y[:i], 'k',linestyle='none', marker = "o", label = "Aufstiegsgeschwindigkeit")
    ax2.plot(np.linspace(0, zeit[i], len(zeit[:i])), fittet_v_y, 'k-', label = "Sinus Fit der Aufstiegsgeschwindigkeit")
    ax2.set_xlabel('Zeit [s]', color='k', fontsize = label_fontsize, fontname = "Arial")
    ax2.set_ylabel('Tropfenaufstiegsgeschwindigkeit [mm/s]', color='k', fontsize = label_fontsize, fontname = "Arial")
    ax2.set_ylim([-40, 80])
    ax2.tick_params('y', colors='k',direction="in")
    ax2.tick_params(direction="in",top=True, right=True, width=1.5, length=8)
    for tick_label in ax2.get_yticklabels():
        tick_label.set_fontsize(label_fontsize)
        tick_label.set_fontname('Arial')  

    for tick_label in ax2.get_xticklabels():
        tick_label.set_fontsize(label_fontsize)
        tick_label.set_fontname('Arial')
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.set_title(erzeuge_titel_für_plots(VERWENDETE_DATEN), fontsize = label_fontsize, fontname = "Arial")
    ax2.legend(lines2, labels2, loc='upper left', fontsize=legend_fontsize)
    fig.tight_layout()
    plt.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\v_y_gekuertzt_"+VERWENDETE_DATEN, dpi = 300)
    plt.close()


#selbe funktion wie plot_rgb_v_y_gekuertzt nur dass der fit nicht in einem gekürten bereich durchgeführt wird und der komplette bereich geplottet wird.
def plot_rgb_v_y():
    fittet_v_y, phase_v_y = fit_funktion(list_v_y, zeit[:-1])
    fittet_rgb, phase_rgb = fit_funktion(rgb_list, zeit[:-1])
    fig, ax1 = plt.subplots(figsize=(16,9))
 
    ax1.plot(zeit[:-1], rgb_list, 'b', linestyle='none', marker = "o", label = "RGB-Farbwert")
    ax1.plot(np.linspace(0, zeit[-1], len(zeit[:-1])), fittet_rgb, 'b-', label = "Sinus Fit von RGB-Farbwert")
    ax1.set_xlabel('Zeit [s]', fontsize = label_fontsize, fontname = "Arial")
    ax1.set_ylabel('RGB-Farbwert', color='b', fontsize = label_fontsize, fontname = "Arial")
    ax1.tick_params('y', colors='b',direction="in")
    ax1.tick_params(direction="in",top=True, right=True, width=1.5, length=8)

    for tick_label in ax1.get_yticklabels():
        tick_label.set_fontsize(label_fontsize)
        tick_label.set_fontname('Arial')  

    for tick_label in ax1.get_xticklabels():
        tick_label.set_fontsize(label_fontsize)
        tick_label.set_fontname('Arial')
        
    ax2 = ax1.twinx()
    ax2.plot(zeit[:-1], list_v_y, 'k',linestyle='none', marker = "o", label = "Aufstiegsgeschwindigkeit")
    ax2.plot(np.linspace(0, zeit[-1], len(zeit[:-1])), fittet_v_y, 'k-', label = "Sinus Fit der Aufstiegsgeschwindigkeit")
    
    ax2.set_ylabel('Tropfenaufstiegsgeschwindigkeit [mm/s]', color='k', fontsize = label_fontsize, fontname = "Arial")
    ax2.set_ylim([-40, 80])
    ax2.tick_params('y', colors='k',direction="in")

    for tick_label in ax2.get_yticklabels():
        tick_label.set_fontsize(label_fontsize)
        tick_label.set_fontname('Arial')  

    for tick_label in ax2.get_xticklabels():
        tick_label.set_fontsize(label_fontsize)
        tick_label.set_fontname('Arial')
        
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.set_title(erzeuge_titel_für_plots(VERWENDETE_DATEN), fontsize = label_fontsize, fontname = "Arial")
    ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=legend_fontsize)
    ax2.tick_params(direction="in",top=True, right=True, width=1.5, length=8)
    
    fig.tight_layout()

    plt.savefig(ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\rgb_v_y_"+VERWENDETE_DATEN, dpi = 300)
    plt.close()

    phasenverschiebung = berechne_phasenverschiebung(fittet_rgb,fittet_v_y)
    pickle.dump(phasenverschiebung, open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\phasenverschiebung", "wb"))
    plot_rgb_v_y_gekuertzt(list_v_y, zeit)


#es wird dei phasenverschiebung zwischen rgb wert und v_y berechnet.
def berechne_phasenverschiebung(datensatz_eins, datensatz_zwei):
    max_eins = max(datensatz_eins)
    min_eins = min(datensatz_eins)
    max_zwei = max(datensatz_zwei)
    min_zwei = min(datensatz_zwei)

    list_null_durchgang_eins = []
    for i in range(len(datensatz_eins)-1):
        if datensatz_eins[i] < (max_eins+min_eins)/2 and datensatz_eins[i + 1] > (max_eins+min_eins)/2:
            list_null_durchgang_eins.append(i)

    list_null_durchgang_zwei = []
    for i in range(len(datensatz_zwei)-1):
        if datensatz_zwei[i] < (max_zwei+min_zwei)/2 and datensatz_zwei[i + 1] > (max_zwei+min_zwei)/2:
            list_null_durchgang_zwei.append(i)
        
    phasenverschiebung = []
    for i in range( 1, min(len(list_null_durchgang_eins), len(list_null_durchgang_zwei)) - 1):
        list = [abs(zeit[list_null_durchgang_eins[i-1]] - zeit[list_null_durchgang_zwei[i]]), abs(zeit[list_null_durchgang_eins[i]] - zeit[list_null_durchgang_zwei[i]]), abs(zeit[list_null_durchgang_eins[i+1]] - zeit[list_null_durchgang_zwei[i]])]
        phasenverschiebung.append(zeit[list_null_durchgang_eins[i + list.index(min(list)) - 1]] - zeit[list_null_durchgang_zwei[i]])
    return phasenverschiebung

#es wird die phasenverschiebung geplottet.
def plot_phasenverschiebung():
    list_phasenverschiebung_in_degree = []
    for VERWENDETE_DATEN in list_verwendete_daten:  
        phasenverschiebung_in_sekunden = np.mean(pickle.load(open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\phasenverschiebung", "rb")))
        phasenverschiebung_in_rad = 2 * np.pi * phasenverschiebung_in_sekunden / 0.8
        phasenverschiebung_in_degree = phasenverschiebung_in_rad * 360 / (2 * np.pi)
        list_phasenverschiebung_in_degree.append(phasenverschiebung_in_degree)

    umbenannte_liste_verwendete_daten = []
    for elem in list_verwendete_daten:
        umbenannte_liste_verwendete_daten.append(erzeuge_titel_für_plots(elem))

    plot_bar(loc_label = "upper left", label = None, pos_string = [], strings = [], x_achse = umbenannte_liste_verwendete_daten, y_achse = list_phasenverschiebung_in_degree, y_modell = None, y_label = 'Phasenverschiebung [°]', x_label = '',title = "Phasenverschiebung", SAVE_DIR =ROOT_DIR+"\\datasets\\output_analyseddata\\phasenverschiebung.png")
    
#hier wird die parametrisierung eines datensatzes durchgeführt. RELEVANT ZUM VERSTEHEN.
def parametrisiere_modell():
    os.mkdir(ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\v_y_positive_negative_all_anzahl_tropfen")
    M_dis = pickle.load(open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\M_dis", "rb"))
    tropfengröße = list(M_dis[:,0])
    step_list = np.round(np.arange(untere_grenze_parametrisierung + step_size, obere_grenze_parametrisierung + step_size, step_size), 2)
    erste_index = tropfengröße.index(step_list[0])
    letzter_index = tropfengröße.index(step_list[-1])
    v_y_gemessen = np.array(M_dis[:,2])[erste_index:letzter_index+1] * 10**-3
    
    guess_schwarmexponent = 1
    p0 = [guess_schwarmexponent]

    fit = curve_fit(get_u_y, step_list* 10**-3, v_y_gemessen, p0=p0)
    pickle.dump(fit[0][0], open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\v_y_positive_negative_all_anzahl_tropfen\\exponent", "wb"))
    return float(fit[0][0])

#hier wird die parametrisierung aller datensätze durchgeführt. RELEVANT ZUM VERSTEHEN.
def gesamt_parametrisierung():
    counter = 0
    step_list = np.round(np.arange(untere_grenze_parametrisierung + step_size, obere_grenze_parametrisierung + step_size, step_size), 2)
    v_y_gemessen_summiert = np.zeros(len(step_list))
    for VERWENDETE_DATEN in list_verwendete_daten:
        if VERWENDETE_DATEN.split("_")[4] != "100":
            M_dis = pickle.load(open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\M_dis", "rb"))
            tropfengröße = list(M_dis[:,0])
            erste_index = tropfengröße.index(step_list[0])
            letzter_index = tropfengröße.index(step_list[-1])
            v_y_gemessen = np.array(M_dis[:,2])[erste_index:letzter_index+1]
            v_y_gemessen_summiert += v_y_gemessen
            counter += 1

    guess_schwarmexponent = 1
    p0 = [guess_schwarmexponent]

    fit = curve_fit(get_u_y, step_list* 10**-3, v_y_gemessen_summiert / counter * 10**-3, p0=p0)
    return float(fit[0][0])

#hier wird der gegebene holdup umgerechnet. Es wird eine größere kontaktzone angenommen.
def umrechnung_holp_up(hold_up):
    Vakt = 0.005            #m^3
    hres_eins = 0.15        #m
    hres_zwei = 0.55        #m
    Vres = (hres_eins + hres_zwei) * np.pi * 0.25 * 0.05**2
    hold_up_neu = (hold_up * Vakt) / (Vakt + Vres)
    
    return hold_up_neu

#hilfsfunktion die hold-up und volumenströme der konti und dispersen phase bereitstellt. und damit die geschwindigkeit der konti phase berechnet.
def get_hold_up_V_x_V_d_u_x(VERWENDETE_DATEN):
    hold_up_ohne_umrechnung = list_konti_dispers_hold_up[VERWENDETE_DATEN][2] #hold_up = epsilon = alpha_y
    hold_up_neu = umrechnung_holp_up(hold_up_ohne_umrechnung) 
    V_x = list_konti_dispers_hold_up[VERWENDETE_DATEN][1] #volume stream of continiuous phase [m^3/s]
    V_d = list_konti_dispers_hold_up[VERWENDETE_DATEN][0] #volume stream of dispersed phase [m^3/s]
    u_x = V_x / (A_col * (1 - hold_up_neu))

    return hold_up_neu, V_x, V_d, u_x


def größenverteilung_bild_zu_bild():
    list_größenverteilung_bild_zu_bild = []

    step_liste = np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2)
    for größe in step_liste:
        list_größenverteilung_bild_zu_bild.append(0)
    
    for bild_index in range(start, ende):
        alle_tropfen_betrachtetes_bild = list_r[bild_index]
        for tropfen in alle_tropfen_betrachtetes_bild:
            height, width, center_y, center_x, durchmesser = morphologie_eines_tropfens(tropfen)

            for index, größe in enumerate(step_liste):
                if durchmesser < größe and durchmesser > größe - step_size:
                    list_größenverteilung_bild_zu_bild[index] = list_größenverteilung_bild_zu_bild[index] + 1

    liste_relative_anzahl_tropfen = []
    for i in range(len(list_größenverteilung_bild_zu_bild)):
        liste_relative_anzahl_tropfen.append(list_größenverteilung_bild_zu_bild[i]/sum(list_größenverteilung_bild_zu_bild))


    tropfengröße, x = np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2), []
    for i in range(len(tropfengröße)):
        if tropfengröße[i] != np.inf:
            string = str(np.round(tropfengröße[i] - step_size, 2)) + " bis " + str(tropfengröße[i])
        else:
            string = str(tropfengröße[i-1])+ " bis " + str(tropfengröße[i])
        x.append(string)
    plot_bar(loc_label = "upper left",label = None, pos_string = None, strings = [], x_achse = x, y_achse = list_größenverteilung_bild_zu_bild, y_modell = None, y_label = "Anzahl Tropfen", x_label = "Tropfengröße [mm]", title = erzeuge_titel_für_plots(VERWENDETE_DATEN), SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\anzahl_tropfen_ohne_ids_"+VERWENDETE_DATEN)

    return list_größenverteilung_bild_zu_bild, liste_relative_anzahl_tropfen

#hier wird berechnet wie lang die folgen aus dic_id sind.
def länge_eine_folge():
    länge_dic, länge_list = {},[]
    os.mkdir(ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\langeids")

    dic_anzahl_laenge_ids_alle_groeßen = {}
    for laenge in range(2,100):
        dic_anzahl_laenge_ids_alle_groeßen[laenge] = 0
    for größe in dic_id_groeßenverteilung:
        länge_dic[größe], summe = [], 0
            
        dic_anzahl_laenge_ids = {}
        for laenge in range(2,100):
            dic_anzahl_laenge_ids[laenge] = 0
            
        for ID in dic_id_groeßenverteilung[größe]:
            laenge_id = dic_id[ID][-1][0] - dic_id[ID][0][0] + 1
            summe += laenge_id
            for laenge in range(2,100):
                if laenge == laenge_id:
                    dic_anzahl_laenge_ids[laenge] = dic_anzahl_laenge_ids[laenge] + 1
                    dic_anzahl_laenge_ids_alle_groeßen[laenge] = dic_anzahl_laenge_ids_alle_groeßen[laenge] + 1 
       
        list_laenge_ids = list(dic_anzahl_laenge_ids)
        anzahl_ids = list(dic_anzahl_laenge_ids.values())
        plot_bar(loc_label = "upper left", label = None, pos_string = None, strings = [], x_achse = list_laenge_ids, y_achse = anzahl_ids, y_modell = None, y_label = "Anzahl ID", x_label = "Länge ID", title = erzeuge_titel_für_plots(VERWENDETE_DATEN), SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\langeids\\langeids_" + str(größe) + "_" + VERWENDETE_DATEN + ".png")

        if len(dic_id_groeßenverteilung[größe]) != 0:
            länge_dic[größe].append(summe/len(dic_id_groeßenverteilung[größe]))
            länge_list.append(summe/len(dic_id_groeßenverteilung[größe]))
        else:
            länge_dic[größe].append(0)
            länge_list.append(0)

    list_laenge_ids = list(dic_anzahl_laenge_ids_alle_groeßen)
    anzahl_ids = list(dic_anzahl_laenge_ids_alle_groeßen.values())
    plot_bar(loc_label = "upper left", label = None, pos_string = None, strings = [], x_achse = list_laenge_ids, y_achse = anzahl_ids, y_modell = None, y_label = "Anzahl ID", x_label = "Länge ID", title = erzeuge_titel_für_plots(VERWENDETE_DATEN), SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\langeids\\langeids_alle_groeßen_" + VERWENDETE_DATEN + ".png") 
    
    tropfengröße, x = np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2), []
    for i in range(len(tropfengröße)):
        if tropfengröße[i] != np.inf:
            string = str(np.round(tropfengröße[i] - step_size, 2)) + " bis " + str(tropfengröße[i])
        else:
            string = str(tropfengröße[i-1])+ " bis " + str(tropfengröße[i])
        x.append(string)

    plot_bar(loc_label = "upper left", label = None, pos_string = None, strings = [], x_achse = x, y_achse = länge_list[0:-1], y_modell = None, y_label = "Durchschnittliche Länge einer ID", x_label = "Tropfengröße [mm]", title = erzeuge_titel_für_plots(VERWENDETE_DATEN), SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\länge_ids_"+VERWENDETE_DATEN)

#hier wird der durchmesser einer kugel mit durchmesser = durchmesser berehcnet.
def Volumen_berechnung_kugel(durchmesser):
    return np.pi / 6 * durchmesser**3

#hier wird der volumenanteil einer größenklasse berechnet.
def Volumenanteil_pro_groeßenklasse():
    v_ges = 0
    for ID in dic_id:
        avg_durchmesser = avg_durchmesser_id(ID)
        v_ges += Volumen_berechnung_kugel(avg_durchmesser)

    v_rel = []
    for größe in dic_id_groeßenverteilung:
        v_größe = 0
        for ID in dic_id_groeßenverteilung[größe]:
            avg_durchmesser = avg_durchmesser_id(ID)
            v_größe += Volumen_berechnung_kugel(avg_durchmesser)

        v_rel.append(v_größe/v_ges)

    tropfengröße, x = np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2), []
    for i in range(len(tropfengröße)):
        if tropfengröße[i] != np.inf:
            string = str(np.round(tropfengröße[i] - step_size, 2)) + " bis " + str(tropfengröße[i])
        else:
            string = str(tropfengröße[i-1])+ " bis " + str(tropfengröße[i])
        x.append(string)
    plot_bar(loc_label = "upper left",label = None, pos_string = None, strings = [], x_achse = x, y_achse = v_rel[:-1], y_modell = None, y_label = "Relativer Volumenanteil", x_label = "Tropfengröße [mm]", title = erzeuge_titel_für_plots(VERWENDETE_DATEN), SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\relativer_volumenanteil_"+VERWENDETE_DATEN)

#hier wird die relative verfolgungsrat je größenklasse berechnet.Vermutlich irrelevant zum verstehen des algorithmus
def relative_verfolgungsrate():
    list_anzahl_verfolgung_in_tropfenklasse = []
    for größe in dic_id_groeßenverteilung:
        anzahl_verfolgung_in_tropfenklasse = 0
        for ID in dic_id_groeßenverteilung[größe]:
            anzahl_verfolgung_in_tropfenklasse += len(dic_id[ID])
        list_anzahl_verfolgung_in_tropfenklasse.append(anzahl_verfolgung_in_tropfenklasse)

    list_relative_verfolgungsrate = np.array(list_anzahl_verfolgung_in_tropfenklasse[:-1])/np.array(list_größenverteilung_bild_zu_bild)
    tropfengröße, x = np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2), []
    for i in range(len(tropfengröße)):
        if tropfengröße[i] != np.inf:
            string = str(np.round(tropfengröße[i] - step_size, 2)) + " bis " + str(tropfengröße[i])
        else:
            string = str(tropfengröße[i-1])+ " bis " + str(tropfengröße[i])
        x.append(string)
    plot_bar(loc_label = "upper left",label = None, pos_string = None, strings = [], x_achse = x, y_achse = list_relative_verfolgungsrate, y_modell = None, y_label = "Verfolgungsrate", x_label = "Tropfengröße [mm]", title = erzeuge_titel_für_plots(VERWENDETE_DATEN), SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\relative_verfolgungsrate_"+VERWENDETE_DATEN)

#hier wird die länge eine datensatzes in miliskunden über die timestamps berechnet. vermutlich irrelevant zum verstehen des algorithmus 
def laenge_datensatz():
    list_zeitdifferenz = []
    for i, VERWENDETE_DATEN in enumerate(list_verwendete_daten):
        list_elem = []
        for elem in os.listdir(ROOT_DIR+"\\datasets\\output\\" + VERWENDETE_DATEN):
            list_elem.append(int(elem))
            
        list_elem.sort()

        timestamp_first = str(list_elem[0])
        timestamp_last = str(list_elem[-1])
                
        zeitdifferenz = (float(timestamp_last[-9:-7]) - float(timestamp_first[-9:-7])) * 60 * 60 * 1000 + (float(timestamp_last[-7:-5]) - float(timestamp_first[-7:-5])) * 60 * 1000 + (float(timestamp_last[-5:-3]) - float(timestamp_first[-5:-3])) * 1000 + float(timestamp_last[-3:]) - float(timestamp_first[-3:])
        list_zeitdifferenz.append(zeitdifferenz)

    umbenannte_liste_verwendete_daten = []
    for elem in list_verwendete_daten:
        umbenannte_liste_verwendete_daten.append(erzeuge_titel_für_plots(elem))

    plot_bar(loc_label = "upper left",label = None, pos_string = None, strings = [], x_achse = umbenannte_liste_verwendete_daten, y_achse = list_zeitdifferenz, y_modell = None, y_label = "Zeit [ms]", x_label = "", title = "Länge der Datensätze", SAVE_DIR = ROOT_DIR+"\\datasets\\output_analyseddata\\länge_datensätze.png")

#hier wird ein plot erstellt, der für jeden zu parametrisierenden datensatz den schwarmexponent dargestellt. vermutlich irrelevant zum verstehen des algorithmus 
def exponent_parametrisierung():
    list_n = []
    list_plot_exponent = []
    for i, VERWENDETE_DATEN in enumerate(list_verwendete_daten):
        if VERWENDETE_DATEN.split("_")[4] != "100":
            n = pickle.load(open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\v_y_positive_negative_all_anzahl_tropfen\\exponent", "rb"))
            list_plot_exponent.append(erzeuge_titel_für_plots(VERWENDETE_DATEN))
            print(n, VERWENDETE_DATEN)
            list_n.append(n)
    plot_bar(loc_label = "upper left",label = None, pos_string = None, strings = [], x_achse = list_plot_exponent, y_achse = list_n, y_modell = None, y_label = "Schwarmexponent", x_label = "", title = "Schwarmexponent je Datensatz", SAVE_DIR = ROOT_DIR+"\\datasets\\output_analyseddata\\exponent_n.png")

#hier wird die zeitdifferenz von bild zu bild berechnet.um im folgenden zb geschwindigkeiten zwischen von tropfen zwischen bildern zu berehnen. RELEVANT ZUM VERSTEHEN
def berechnezeit_differenz():
    if konstante_framerate == False:
        list_zeitdifferenz, sum_time, zeit = [], 0, [0]
        
        for i in range(1, len(list_elem)):
            timestamp_first = str(list_elem[i-1])
            timestamp_last = str(list_elem[i])
            
            zeitdifferenz = (float(timestamp_last[-9:-7]) - float(timestamp_first[-9:-7])) * 60 * 60 * 1000 + (float(timestamp_last[-7:-5]) - float(timestamp_first[-7:-5])) * 60 * 1000 + (float(timestamp_last[-5:-3]) - float(timestamp_first[-5:-3])) * 1000 + float(timestamp_last[-3:]) - float(timestamp_first[-3:])
            list_zeitdifferenz.append(zeitdifferenz)
            zeit.append(np.round(zeit[i-1] + zeitdifferenz/1000, 4))

        plot_funktion(np.linspace(0, len(zeit[:-1]), len(zeit[:-1])), [list_zeitdifferenz], "Zeitdifferenz zum vorherigen Bild [ms]", "Bild_Index", ROOT_DIR + "\\datasets\\timestamps\\timestamps_" + VERWENDETE_DATEN + ".png", None, None, erzeuge_titel_für_plots(VERWENDETE_DATEN))
        plot_funktion(np.linspace(0, len(zeit), len(zeit)), [zeit, np.linspace(0, 10, len(zeit)) ], "Zeit in [s]", "Bild_Index", ROOT_DIR + "\\datasets\\timestamps\\zeit_" + VERWENDETE_DATEN + ".png", None, None, erzeuge_titel_für_plots(VERWENDETE_DATEN))
        
        return zeit
    else:
        timestamp_first = str(list_elem[0])
        timestamp_last = str(list_elem[-1])
        länge_datensatz = len(list_elem)
        zeitdifferenz = (float(timestamp_last[-9:-7]) - float(timestamp_first[-9:-7])) * 60 * 60 * 1000 + (float(timestamp_last[-7:-5]) - float(timestamp_first[-7:-5])) * 60 * 1000 + (float(timestamp_last[-5:-3]) - float(timestamp_first[-5:-3])) * 1000 + float(timestamp_last[-3:]) - float(timestamp_first[-3:])
        
        return np.linspace(0, zeitdifferenz/1000, len(list_elem))

#hier wird die relative anzahl an tropfen je größenklasse berechnet.
def berechne_verhältnis_anzahl_tropfen():
    M_dis = pickle.load(open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\M_dis", "rb"))
    step_list = list(np.round(np.arange(untere_grenze + step_size, obere_grenze + step_size, step_size), 2))
    anzahl_tropfen, x = list(M_dis[:,1]), []
    for i, größe in enumerate(step_list):
        x.append(str(np.round(größe - step_size, 2)) + " bis " + str(größe))
        
    verhältnis = np.array(list_größenverteilung_bild_zu_bild)/np.array(anzahl_tropfen[:-1])
    plot_bar(loc_label = "upper left",label = None, pos_string = None, strings = [], x_achse = x, y_achse = verhältnis, y_modell = None, y_label = "anzahl durch anzahl", x_label = "Tropfengröße [mm]", title = erzeuge_titel_für_plots(VERWENDETE_DATEN), SAVE_DIR = ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN + "\\anzahl_durch_anzahl"+VERWENDETE_DATEN)


modus = int(input("modus = 0: Tropfenerkennung für Datensatz, modus = 1: Tropfenverfolgung - erstellung von dic_id (Vorrausetzung modus 0 muss schon einmal gelaufen sein für den Datensatz), modus = 2: Datenanalyse (Vorrausetzung: modus 0 und 1 muss schon gelaufen sein für den Datensatz)"))
if modus == 0:
    list_modus = [0]
if modus == 1:
    list_modus = [1]
if modus == 2:
    list_modus = [2]


if __name__ == "__main__":
    if 0 in list_modus:
        tropfen_erkennen(ROOT_DIR)
    if 1 in list_modus:
        for i, VERWENDETE_DATEN in enumerate(list_verwendete_daten):
            list_r, list_elem, ende = load_data()
            zeit = berechnezeit_differenz()
            list_r, list_elem, zeit = list_r[start:ende+1], list_elem[start:ende+1], zeit[start:ende+1]
            dic_id = konsekutive_bilderverfolgung()
    if 2 in list_modus:
        for i, VERWENDETE_DATEN in enumerate(list_verwendete_daten):
            if VERWENDETE_DATEN.split("_")[4] != "100":
                hold_up, V_x, V_d, u_x = get_hold_up_V_x_V_d_u_x(VERWENDETE_DATEN)
            
            list_r, list_elem, ende = load_data()
            zeit = berechnezeit_differenz()
            list_r, list_elem, zeit = list_r[start:ende+1], list_elem[start:ende+1], zeit[start:ende+1]
    
            dic_id = pickle.load(open(ROOT_DIR+"\\datasets\\daten\\dic_id_"+VERWENDETE_DATEN, "rb"))
            dic_id_groeßenverteilung = nach_groeße_sortieren()

            try:
                os.mkdir(ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN)
            except:
                if remove_data == True:
                    shutil.rmtree(ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN)
                    os.mkdir(ROOT_DIR + "\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN)
                else:
                    sys.exit("Datei gibt es schon")
                
            Volumenanteil_pro_groeßenklasse()
            länge_eine_folge()
            size_distribution()
            rgb_list = get_rgb()
            list_v_x, list_v_y = geschwindigkeit_bild_index_bis_bild_index_plus_eins()
            plot_rgb_v_y()
            
            stationaerer_sauterdurchmesser()
            verhaeltis_erkannte_bilder()
            list_größenverteilung_bild_zu_bild, liste_relative_anzahl_tropfen = größenverteilung_bild_zu_bild()
            liste_relative_anzahl_id = größenverteilung_id()
            plot_modell_wiederholder(liste_relative_anzahl_id,liste_relative_anzahl_tropfen)
            relative_verfolgungsrate()
            berechne_verhältnis_anzahl_tropfen()
        n_ges = gesamt_parametrisierung()
        for i, VERWENDETE_DATEN in enumerate(list_verwendete_daten):
            if VERWENDETE_DATEN.split("_")[4] != "100":
                hold_up, V_x, V_d, u_x = get_hold_up_V_x_V_d_u_x(VERWENDETE_DATEN)
            
            list_r, list_elem, ende = load_data()
            zeit = berechnezeit_differenz()
            list_r, list_elem, zeit = list_r[start:ende+1], list_elem[start:ende+1], zeit[start:ende+1]

            dic_id = pickle.load(open(ROOT_DIR+"\\datasets\\daten\\dic_id_"+VERWENDETE_DATEN, "rb"))
            dic_id_groeßenverteilung = pickle.load(open(ROOT_DIR+"\\datasets\\daten\\dic_id_groeßenverteilung_"+VERWENDETE_DATEN, "rb"))
            M_dis = pickle.load(open(ROOT_DIR+"\\datasets\\output_analyseddata\\"+VERWENDETE_DATEN+"\\M_dis", "rb"))
            
            n = parametrisiere_modell()
            plot_size_distribution()
            AnzahlderTropfen_Geschwindigkeit_RGB()

            make_video_normalverteilung(1,ROOT_DIR+"\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN+"\\AnzahlderTropfen_Geschwindigkeit_RGB",ROOT_DIR+"\\datasets\\output_analyseddata\\" + VERWENDETE_DATEN+"\\AnzahlderTropfen_Geschwindigkeit_RGB\\video.mp4")
        laenge_datensatz()
        exponent_parametrisierung()
        plot_phasenverschiebung()
        plot_unendliche_aufstigsgeschwindigkeit()
