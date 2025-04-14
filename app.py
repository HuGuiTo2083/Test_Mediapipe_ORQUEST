from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp

from mss import mss
import cv2
import numpy as np
from flask_socketio import SocketIO, emit
from datetime import datetime  
from skimage.metrics import structural_similarity as ssim

import os
port = int(os.environ.get("PORT", 5000))


mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
# ---- instancias persistentes ----






# mp.solutions.drawing_utils contiene funciones para dibujar los landmarks
# (puntos) y las conexiones (líneas) en la imagen.
mp_drawing = mp.solutions.drawing_utils

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tu_clave_secreta'  # Cambia esto en producción!
socketio = SocketIO(app, cors_allowed_origins="*")


# Configuración de captura de pantalla
MONITOR_NUMBER = 1  # Usar sct.monitors para ver opciones
FRAME_RATE = 10     # Cuadros por segundo
# Constantes críticas
OCR_CONFIG = '--psm 11 --oem 3 -c preserve_interword_spaces=1'
# ================================
# 1) Configuración de MediaPipe
# ================================
# mp.solutions.hands es un módulo de MediaPipe especializado en la detección y
# rastreo de manos. Nos provee la clase "Hands" para hacer el análisis.



import math

def log_event(message):
    print(message)
    socketio.emit("gesture_event",              # ←  se envía al front‑end
                  {"message": message})  
    # Emite el evento "gesture_event" a todos los clientes conectados
    #socketio.emit("gesture_event", {"message": message})



def gen_frames(camera_index):

    """
    Esta función genera un flujo (stream) de fotogramas (frames) que el navegador
    irá recibiendo en tiempo real. Dentro de ella:
      - Capturamos la cámara con OpenCV
      - Procesamos con MediaPipe
      - Dibujamos (opcional) los resultados
      - Codificamos en JPEG y enviamos cada frame al navegador
    """

    # Crea el objeto de captura para la cámara (0 = cámara principal)
    cap = cv2.VideoCapture(camera_index)
    
    # Ajustes opcionales de la cámara (descomentarlos si quieres forzar un tamaño)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # =========================================
    # 2) Instancia del detector de manos
    # =========================================
    # Hands(...) permite configurar ciertos parámetros, entre ellos:
    #   - model_complexity: cuán complejo es el modelo (0, 1 o 2), afectando
    #     precisión y rendimiento.
    #   - min_detection_confidence: confianza mínima para que MediaPipe
    #     considere que hay una mano detectada.
    #   - min_tracking_confidence: confianza mínima para el rastreo continuo.
    #
    # El "with ... as hands:" se asegura de cerrar limpiamente el recurso.
    #

    with mp_hands.Hands(
     static_image_mode=False,        # Para video, False (activar tracking)
        max_num_hands=2,               # Número máximo de manos
        model_complexity=1,            # Complejidad del modelo (0,1,2)
        min_detection_confidence=0.5,  # Confianza mínima para detectar manos
        min_tracking_confidence=0.5    # Confianza mínima para el tracking
    )as hands, mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose, mp_face_mesh.FaceMesh(
             max_num_faces=1,              # Detectamos 1 cara
             refine_landmarks=True,        # Más precisión en ojos/labios
             min_detection_confidence=0.5,
             min_tracking_confidence=0.5
    ) as face_mesh:
        
        while True:
            # Capturamos frame a frame de la cámara
            success, frame = cap.read()
            if not success:
                break  # si falla, salimos del bucle

            # Volteamos horizontalmente para un efecto espejo (opcional)
            frame = cv2.flip(frame, 1)
            
            # Convertir de BGR -> RGB, porque MediaPipe trabaja mejor en RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # =========================================
            # 3) Procesar la imagen con MediaPipe
            # =========================================
            # Esto hace el análisis de la mano. Devuelve un objeto "results"
            # con información sobre las manos detectadas.
            results = hands.process(frame_rgb)
            results_pose = pose.process(frame_rgb)
            results_face = face_mesh.process(frame_rgb)

            # Volvemos a convertir a BGR para dibujar con OpenCV
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Dimensiones de la imagen
            h, w, _ = frame.shape


            # ================================================
            # 4) Revisar si hay manos y (opcional) dibujarlas
            # ================================================
            #PARA LA MANO---
            if results.multi_hand_landmarks:
                # results.multi_hand_landmarks es una lista. Cada elemento
                # corresponde a una mano detectada y trae 21 landmarks (puntos).
                for hand_landmarks in results.multi_hand_landmarks:
                    
                    # (Opcional) Dibuja puntos y conexiones
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    
                    # =================================
                    # A) Dibujar bounding box (recuadro)
                    # =================================
                    h, w, _ = frame.shape

                    # Extraemos las coordenadas x,y de cada landmark:
                    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # Dibujamos el rectángulo
                    cv2.rectangle(
                        frame,
                        (x_min, y_min),
                        (x_max, y_max),
                        (0, 255, 0),
                        2
                    )

                    # =================================
                    # B) Lógica para detectar gestos
                    # =================================
                    # Aquí podríamos hacer cualquier análisis: por ejemplo
                    # si se levanta cierto dedo, si la mano está cerrada, etc.
                    
                    # MediaPipe indexa los 21 puntos (landmarks) de la mano de la siguiente forma:
                    #if :
                    #
                    #  0: Wrist
                    #     - Español: Muñeca
                    #     - Para acceder en MediaPipe: mp_hands.HandLandmark.WRIST
                    #
                    #  1: Thumb CMC
                    #     - Español: Articulación carpometacarpiana del pulgar
                    #     - Para acceder en MediaPipe: mp_hands.HandLandmark.THUMB_CMC
                    #
                    #  2: Thumb MCP
                    #     - Español: Articulación metacarpofalángica del pulgar
                    #     - Para acceder en MediaPipe: mp_hands.HandLandmark.THUMB_MCP
                    #
                    #  3: Thumb IP
                    #     - Español: Articulación interfalángica del pulgar
                    #     - Para acceder en MediaPipe: mp_hands.HandLandmark.THUMB_IP
                    #
                    #  4: Thumb Tip
                    #     - Español: Punta del pulgar
                    #     - Para acceder en MediaPipe: mp_hands.HandLandmark.THUMB_TIP
                    #
                    #  5: Index Finger MCP
                    #     - Español: Articulación metacarpofalángica del dedo índice
                    #     - Para acceder en MediaPipe: mp_hands.HandLandmark.INDEX_FINGER_MCP
                    #
                    #  6: Index Finger PIP
                    #     - Español: Articulación interfalángica proximal del dedo índice
                    #     - Para acceder en MediaPipe: mp_hands.HandLandmark.INDEX_FINGER_PIP
                    #
                    #  7: Index Finger DIP
                    #     - Español: Articulación interfalángica distal del dedo índice
                    #     - Para acceder en MediaPipe: mp_hands.HandLandmark.INDEX_FINGER_DIP
                    #
                    #  8: Index Finger Tip
                    #     - Español: Punta del dedo índice
                    #     - Para acceder en MediaPipe: mp_hands.HandLandmark.INDEX_FINGER_TIP
                    #
                    #  9: Middle Finger MCP
                    #     - Español: Articulación metacarpofalángica del dedo medio
                    #     - Para acceder en MediaPipe: mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                    #
                    # 10: Middle Finger PIP
                    #     - Español: Articulación interfalángica proximal del dedo medio
                    #     - Para acceder en MediaPipe: mp_hands.HandLandmark.MIDDLE_FINGER_PIP
                    #
                    # 11: Middle Finger DIP
                    #     - Español: Articulación interfalángica distal del dedo medio
                    #     - Para acceder en MediaPipe: mp_hands.HandLandmark.MIDDLE_FINGER_DIP
                    #
                    # 12: Middle Finger Tip
                    #     - Español: Punta del dedo medio
                    #     - Para acceder en MediaPipe: mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                    #
                    # 13: Ring Finger MCP (el más cerxcano a la palma)
                    #     - Español: Articulación metacarpofalángica del dedo anular
                    #     - Para acceder en MediaPipe: mp_hands.HandLandmark.RING_FINGER_MCP
                    #
                    # 14: Ring Finger PIP
                    #     - Español: Articulación interfalángica proximal del dedo anular
                    #     - Para acceder en MediaPipe: mp_hands.HandLandmark.RING_FINGER_PIP
                    #
                    # 15: Ring Finger DIP
                    #     - Español: Articulación interfalángica distal del dedo anular
                    #     - Para acceder en MediaPipe: mp_hands.HandLandmark.RING_FINGER_DIP
                    #
                    # 16: Ring Finger Tip (la punta, la uña prácticamente)
                    #     - Español: Punta del dedo anular
                    #     - Para acceder en MediaPipe: mp_hands.HandLandmark.RING_FINGER_TIP
                    #
                    # 17: Pinky MCP
                    #     - Español: Articulación metacarpofalángica del dedo meñique
                    #     - Para acceder en MediaPipe: mp_hands.HandLandmark.PINKY_MCP
                    #
                    # 18: Pinky PIP
                    #     - Español: Articulación interfalángica proximal del dedo meñique
                    #     - Para acceder en MediaPipe: mp_hands.HandLandmark.PINKY_PIP
                    #
                    # 19: Pinky DIP
                    #     - Español: Articulación interfalángica distal del dedo meñique
                    #     - Para acceder en MediaPipe: mp_hands.HandLandmark.PINKY_DIP
                    #
                    # 20: Pinky Tip
                    #     - Español: Punta del dedo meñique
                    #     - Para acceder en MediaPipe: mp_hands.HandLandmark.PINKY_TIP



                    # Ejemplo: extraer el Tip del dedo índice
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    # b) Comparamos la posición del dedo índice con la muñeca (por ejemplo)
                    #    para detectar si está "levantado" (arriba)
                    wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                    index_tip_y = index_finger_tip.y

                    # Recuerda que .y está normalizado (0 = parte superior, 1 = parte inferior de la imagen)
                    # Por ende, un índice menor de y significa "más arriba" en la imagen.
                    # Si la punta del índice está por encima de la muñeca, interpretamos "dedo levantado".
                    
                    #if index_tip_y < wrist_y:
                        #print("Dedo índice levantado")


                    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
                    dist_pinky = math.sqrt((pinky_tip.x - pinky_mcp.x)**2 + (pinky_tip.y - pinky_mcp.y)**2)
                    #print("Coordenada de pinky tip: " + str(pinky_tip))
                    #print("coordenada de pinky mcp: " + str(pinky_mcp))

                    # print("distancia entre pukye es: " + str(dist_pinky))
                    # if dist_pinky < 0.1:
                    #     print("Pinky tip está junto con pinky mcp")
                    #     print("---------------------------")

                    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    ring_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
                    dist_ring_finger = math.sqrt((ring_finger_tip.x - ring_finger_mcp.x)**2 + (ring_finger_tip.y - ring_finger_mcp.y)**2)

                    # print("distancia entre ring fingers es: " + str(dist_ring_finger))

                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    dist_thumb_and_index = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)

                    # print("distancia entre thumb e index: " + str(dist_thumb_and_index))
                    
                    # if dist_thumb_and_index < 0.1:
                    #     print("se juntó el pulgar con el indice---")


                    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
                    dist_pinky_pip_and_ring_tip = math.sqrt((ring_finger_tip.x - pinky_pip.x)**2 + (ring_finger_tip.y - pinky_pip.y)**2)

                    # print("distancia entre pynky_pip y ring_ringer_tip: " + str(dist_pinky_pip_and_ring_tip))
                    
                    # if dist_pinky_pip_and_ring_tip < 0.1:
                    #     print("---se juntó el anular con el meñique " )
                        
                    ring_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
                    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                    dist_ring_finger_pip_and_middle_tip = math.sqrt((ring_finger_pip.x - middle_finger_tip.x)**2 + (ring_finger_pip.y - middle_finger_tip.y)**2)

                    # print("distancia entre ring_finger_pip y middle_finger_tip: " + str(dist_ring_finger_pip_and_middle_tip))

                    # if dist_ring_finger_pip_and_middle_tip < 0.1:
                    #     print("se juntó el mediano con el anular * * *" )

                    if dist_thumb_and_index < 0.1 and dist_pinky_pip_and_ring_tip < 0.1 and dist_ring_finger_pip_and_middle_tip < 0.1:
                       # print()
                        log_event("---- SE ESTÁ ESCRIBIENDO ----" + str(datetime.now()))
                    # c) Podrías detectar un puño cerrado si todos los tips (Thumb Tip, Index Tip, etc.)
                    #    están por debajo de sus MCP, o si la distancia entre ellos y la palma es mínima, etc.
                    #
                    # Por ejemplo (muy simplificado), para ver si todos los dedos están por debajo
                    # de un cierto valor (simbolizando un puño):
                    finger_tips_indices = [4, 8, 12, 16, 20]  # pulgar, índice, medio, anular, meñique
                    # Convertimos a un array:
                    finger_tips_positions = [hand_landmarks.landmark[i] for i in finger_tips_indices]

                    # Chequeo sencillo: si todos los tips están por debajo (y mayor que) la muñeca
                    # (es decir más cerca de la parte inferior de la imagen, = puño cerrado).

                    # if all(tip.y > wrist_y for tip in finger_tips_positions):
                    #     print("Puño cerrado")

                    # NOTA: Para lógica de gestos más robusta, deberás revisar distancias o ángulos
                    # entre articulaciones, pero esto te da la idea general.

            # Para pose
            # DONDE SE USA CODO, HOMBRO Y MUÑECA
            if results_pose.pose_landmarks:
                # dibujar o analizar hombros/codos
                mp_drawing.draw_landmarks(
                    frame,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
                landmarks_pose = results_pose.pose_landmarks.landmark

                #  - Right Wrist (16)
                #  - Left Elbow (13)
                #  - Left Wrist (15)
                #  - Right Elbow (14)
                right_wrist = landmarks_pose[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                left_elbow = landmarks_pose[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                left_wrist = landmarks_pose[mp_pose.PoseLandmark.LEFT_WRIST.value]
                right_elbow = landmarks_pose[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                nose = landmarks_pose[mp_pose.PoseLandmark.NOSE.value]

                if (right_elbow.y < nose.y) and (right_wrist.y < right_elbow.y):
                    #print("------BRAZO DERECHO LEVANTADO-------" + str(datetime.now()))
                    log_event("------BRAZO DERECHO LEVANTADO-------" + str(datetime.now()))

                if (left_elbow.y < nose.y) and (left_wrist.y < left_elbow.y):
                    #print("------BRAZO IZQUIERDO LEVANTADO-----" + str(datetime.now()))
                    log_event("------BRAZO IZQUIERDO LEVANTADO-----" + str(datetime.now()))
                                # Calculamos distancias Euclidianas en 2D (x,y)
                # 1) Distancia: muñeca derecha vs. codo izquierdo
                dist_rw_le = math.sqrt(
                    (right_wrist.x - left_elbow.x)**2 +
                    (right_wrist.y - left_elbow.y)**2
                )

                # print("distancia entre muñeca derecha y codo izquierdo" + str(dist_rw_le))

                # 2) Distancia: muñeca izquierda vs. codo derecho
                dist_lw_re = math.sqrt(
                    (left_wrist.x - right_elbow.x)**2 +
                    (left_wrist.y - right_elbow.y)**2
                )

                # print("distancia entre muñeca ixquierda y codo derecho" + str(dist_lw_re))


                # Definimos un umbral en coordenadas normalizadas
                threshold = 0.2  # Ajusta según lo que consideres “cerca”

                # if dist_rw_le < threshold:
                #     print("Muñeca derecha CERCA del codo izquierdo!")

                # if dist_lw_re < threshold:
                #     print("Muñeca izquierda CERCA del codo derecho!")

                if dist_rw_le < threshold and dist_lw_re < threshold:
                    #print("--- SE ESTÁ CRUZANDO DE BRAZOS ---" + str(datetime.now()))
                    log_event("--- SE ESTÁ CRUZANDO DE BRAZOS ---" + str(datetime.now()))

             # -------------------------
            #  LÓGICA DE FACE MESH
            # -------------------------
            #ESCLUSIVAMENTE DE LA CARA
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    # (Opcional) Dibuja la malla de la cara
                    # mp_drawing.draw_landmarks(
                    #     frame,
                    #     face_landmarks,
                    #     mp_face_mesh.FACEMESH_CONTOURS
                    # )

                    # Extraemos puntos para los párpados
                    # Ojo izquierdo
                    left_eye_top = face_landmarks.landmark[159]
                    left_eye_bottom = face_landmarks.landmark[145]
                    dist_left_eye = math.sqrt(
                        (left_eye_top.x - left_eye_bottom.x)**2 +
                        (left_eye_top.y - left_eye_bottom.y)**2
                    )

                    # print("distancia ojo izquierdo" + str(dist_left_eye))

                    # Ojo derecho
                    right_eye_top = face_landmarks.landmark[386]
                    right_eye_bottom = face_landmarks.landmark[374]
                    dist_right_eye = math.sqrt(
                        (right_eye_top.x - right_eye_bottom.x)**2 +
                        (right_eye_top.y - right_eye_bottom.y)**2
                    )

                    # Umbral para considerar "ojo cerrado"
                    threshold = 0.01

                    # Verificamos cada ojo
                    ojo_izq_cerrado = dist_left_eye < threshold
                    ojo_der_cerrado = dist_right_eye < threshold

                    if ojo_izq_cerrado and ojo_der_cerrado:
                        #print("Ambos ojos cerrados   " + str(datetime.now()))
                        log_event("Ambos ojos cerrados   " + str(datetime.now()))
                    elif ojo_izq_cerrado:
                        #print("Ojo izquierdo cerrado  " + str(datetime.now()))
                        log_event("Ojo izquierdo cerrado  " + str(datetime.now()))
                    elif ojo_der_cerrado:
                        #print("Ojo derecho cerrado  " + str(datetime.now()))
                        log_event("Ojo derecho cerrado  " + str(datetime.now()))
                    else:
                        # Opcional: no imprimir nada o imprimir "ojos abiertos"
                        pass

            # LÓGICA PARA DETECTAR MANO EN LA BARBILLA
            if results_face.multi_face_landmarks and results.multi_hand_landmarks:
                # Normalmente solo tenemos 1 cara (max_num_faces=1),
                # pero por seguridad iteramos. (o tomamos la primera cara directamente)
                for face_landmarks in results_face.multi_face_landmarks:
                    # Landmark de la barbilla (en Face Mesh, índice 152 suele estar cerca del mentón)
                    chin_landmark = face_landmarks.landmark[152]
                    chin_x, chin_y, chin_z = chin_landmark.x, chin_landmark.y, chin_landmark.z

                    # Podrías iterar sobre las manos detectadas
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Ejemplo: usar la muñeca para aproximar la posición de la mano
                        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

                        # Calcular la distancia Euclidiana en 3D:
                        dist_chin_hand = math.sqrt(
                            (chin_x - wrist_x)**2 + 
                            (chin_y - wrist_y)**2 + 
                            (chin_z - wrist_z)**2
                        )

                        # print("la distancia entre la muñeca y la barbilla es: " + str(dist_chin_hand))

                        # Escoge un umbral en coordenadas normalizadas
                        # (a veces ~0.05-0.1 puede servir, depende de la cámara/distancia)
                        threshold = 0.16

                        if dist_chin_hand < threshold:
                            #print("------- MANO SOBRE LA BARBILLA ---------" + str(datetime.now()))
                            log_event("------- MANO SOBRE LA BARBILLA ---------" + str(datetime.now()))


            # Codifica el fotograma como JPEG y envía la secuencia de bytes
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # yield: Genera un bloque de datos con el formato multipart/x-mixed-replace
            # para que el navegador actualice el <img> en tiempo real
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



# Configuración de OpenRouter (https://openrouter.ai/)
OPENROUTER_API_KEY = "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"  # Reemplazar con tu clave
DEEPSEEK_MODEL = "deepseek/deepseek-r1:free"
PROMPT_IA = """Analiza esta diapositiva siguiendo estas reglas:
1. IGNORA COMPLETAMENTE: 
   - Barras de herramientas/menús (Archivo, Editar, Ver)
   - Números de página/palabras
   - Botones de interfaz (X, Minimizar, etc.)
   - Cualquier elemento fuera del área central de contenido

2. ENFÓCATE EN:
   - Títulos con tamaño de fuente grande
   - Listas con viñetas/numeración
   - Diagramas o gráficos centrales
   - Texto en cuadros/recuadros destacados

3. Devuelve en JSON:
{
   "titulo": "Texto del título principal",
   "puntos_clave": ["punto 1", "punto 2", "punto 3"],
   "elementos_graficos": ["tipo de gráfico detectado"]
}"""


# Configuración adicional
# Actualiza el diccionario de aplicaciones
PRESENTATION_APPS = {
    'powerpoint': [
        'powerpoint', 
        'ppt', 
        'ppsx', 
        'presentación',
        'presentation',
        'slide show'
    ],
    'libreoffice': [
        'libreoffice writer',  # Agregado a petición
        'libreoffice impress',
        'odp',
        'presentación',
        'slide'
    ],
    'pdf': [
        'adobe reader',
        'foxit',
        'pdf-xchange',
        'visor pdf'
    ],
    'canva': [
        'canva',
        'diseño'
    ],
    'web': [
        'google slides',
        'prezi',
        'genially'
    ]
}



def listar_camaras_disponibles(max_test=10):
    """Devuelve una lista de índices de cámaras disponibles."""
    camaras = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            camaras.append(i)
        cap.release()
    return camaras


@app.route('/')
def index():
    indices_camaras = listar_camaras_disponibles()
    return render_template('index.html', cameras=indices_camaras)



@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    return Response(gen_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/upload', methods=['POST'])
def upload():
    global last_frame

    # 1. decodificar JPEG recibido
    jpg = np.frombuffer(request.data, dtype=np.uint8)
    frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    if frame is None:
        return ('', 204)

    # 2. procesar (copia/pega tu lógica; aquí muy resumida)
    frame = cv2.flip(frame, 1)
    frame_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    
    # Volteamos horizontalmente para un efecto espejo (opcional)
    
    
    # =========================================
    # 3) Procesar la imagen con MediaPipe
    # =========================================
    # Esto hace el análisis de la mano. Devuelve un objeto "results"
    # con información sobre las manos detectadas.

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands_temp, mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose_temp, mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_temp:
        results = hands_temp.process(frame_rgb)
        results_pose = pose_temp.process(frame_rgb)
        results_face = face_temp.process(frame_rgb)

    # Volvemos a convertir a BGR para dibujar con OpenCV
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    # Dimensiones de la imagen
    h, w, _ = frame.shape
    # ================================================
    # 4) Revisar si hay manos y (opcional) dibujarlas
    # ================================================
    #PARA LA MANO---
    if results.multi_hand_landmarks:
        # results.multi_hand_landmarks es una lista. Cada elemento
        # corresponde a una mano detectada y trae 21 landmarks (puntos).
        for hand_landmarks in results.multi_hand_landmarks:
            
            # (Opcional) Dibuja puntos y conexiones
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            
            # =================================
            # A) Dibujar bounding box (recuadro)
            # =================================
            h, w, _ = frame.shape
            # Extraemos las coordenadas x,y de cada landmark:
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Dibujamos el rectángulo
            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )
            # =================================
            # B) Lógica para detectar gestos
            # =================================
            # Aquí podríamos hacer cualquier análisis: por ejemplo
            # si se levanta cierto dedo, si la mano está cerrada, etc.
            
            # MediaPipe indexa los 21 puntos (landmarks) de la mano de la siguiente forma:
            #if :
            #
            #  0: Wrist
            #     - Español: Muñeca
            #     - Para acceder en MediaPipe: mp_hands.HandLandmark.WRIST
            #
            #  1: Thumb CMC
            #     - Español: Articulación carpometacarpiana del pulgar
            #     - Para acceder en MediaPipe: mp_hands.HandLandmark.THUMB_CMC
            #
            #  2: Thumb MCP
            #     - Español: Articulación metacarpofalángica del pulgar
            #     - Para acceder en MediaPipe: mp_hands.HandLandmark.THUMB_MCP
            #
            #  3: Thumb IP
            #     - Español: Articulación interfalángica del pulgar
            #     - Para acceder en MediaPipe: mp_hands.HandLandmark.THUMB_IP
            #
            #  4: Thumb Tip
            #     - Español: Punta del pulgar
            #     - Para acceder en MediaPipe: mp_hands.HandLandmark.THUMB_TIP
            #
            #  5: Index Finger MCP
            #     - Español: Articulación metacarpofalángica del dedo índice
            #     - Para acceder en MediaPipe: mp_hands.HandLandmark.INDEX_FINGER_MCP
            #
            #  6: Index Finger PIP
            #     - Español: Articulación interfalángica proximal del dedo índice
            #     - Para acceder en MediaPipe: mp_hands.HandLandmark.INDEX_FINGER_PIP
            #
            #  7: Index Finger DIP
            #     - Español: Articulación interfalángica distal del dedo índice
            #     - Para acceder en MediaPipe: mp_hands.HandLandmark.INDEX_FINGER_DIP
            #
            #  8: Index Finger Tip
            #     - Español: Punta del dedo índice
            #     - Para acceder en MediaPipe: mp_hands.HandLandmark.INDEX_FINGER_TIP
            #
            #  9: Middle Finger MCP
            #     - Español: Articulación metacarpofalángica del dedo medio
            #     - Para acceder en MediaPipe: mp_hands.HandLandmark.MIDDLE_FINGER_MCP
            #
            # 10: Middle Finger PIP
            #     - Español: Articulación interfalángica proximal del dedo medio
            #     - Para acceder en MediaPipe: mp_hands.HandLandmark.MIDDLE_FINGER_PIP
            #
            # 11: Middle Finger DIP
            #     - Español: Articulación interfalángica distal del dedo medio
            #     - Para acceder en MediaPipe: mp_hands.HandLandmark.MIDDLE_FINGER_DIP
            #
            # 12: Middle Finger Tip
            #     - Español: Punta del dedo medio
            #     - Para acceder en MediaPipe: mp_hands.HandLandmark.MIDDLE_FINGER_TIP
            #
            # 13: Ring Finger MCP (el más cerxcano a la palma)
            #     - Español: Articulación metacarpofalángica del dedo anular
            #     - Para acceder en MediaPipe: mp_hands.HandLandmark.RING_FINGER_MCP
            #
            # 14: Ring Finger PIP
            #     - Español: Articulación interfalángica proximal del dedo anular
            #     - Para acceder en MediaPipe: mp_hands.HandLandmark.RING_FINGER_PIP
            #
            # 15: Ring Finger DIP
            #     - Español: Articulación interfalángica distal del dedo anular
            #     - Para acceder en MediaPipe: mp_hands.HandLandmark.RING_FINGER_DIP
            #
            # 16: Ring Finger Tip (la punta, la uña prácticamente)
            #     - Español: Punta del dedo anular
            #     - Para acceder en MediaPipe: mp_hands.HandLandmark.RING_FINGER_TIP
            #
            # 17: Pinky MCP
            #     - Español: Articulación metacarpofalángica del dedo meñique
            #     - Para acceder en MediaPipe: mp_hands.HandLandmark.PINKY_MCP
            #
            # 18: Pinky PIP
            #     - Español: Articulación interfalángica proximal del dedo meñique
            #     - Para acceder en MediaPipe: mp_hands.HandLandmark.PINKY_PIP
            #
            # 19: Pinky DIP
            #     - Español: Articulación interfalángica distal del dedo meñique
            #     - Para acceder en MediaPipe: mp_hands.HandLandmark.PINKY_DIP
            #
            # 20: Pinky Tip
            #     - Español: Punta del dedo meñique
            #     - Para acceder en MediaPipe: mp_hands.HandLandmark.PINKY_TIP
            # Ejemplo: extraer el Tip del dedo índice
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            # b) Comparamos la posición del dedo índice con la muñeca (por ejemplo)
            #    para detectar si está "levantado" (arriba)
            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            index_tip_y = index_finger_tip.y
            # Recuerda que .y está normalizado (0 = parte superior, 1 = parte inferior de la imagen)
            # Por ende, un índice menor de y significa "más arriba" en la imagen.
            # Si la punta del índice está por encima de la muñeca, interpretamos "dedo levantado".
            
            #if index_tip_y < wrist_y:
                #print("Dedo índice levantado")
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
            dist_pinky = math.sqrt((pinky_tip.x - pinky_mcp.x)**2 + (pinky_tip.y - pinky_mcp.y)**2)
            #print("Coordenada de pinky tip: " + str(pinky_tip))
            #print("coordenada de pinky mcp: " + str(pinky_mcp))
            # print("distancia entre pukye es: " + str(dist_pinky))
            # if dist_pinky < 0.1:
            #     print("Pinky tip está junto con pinky mcp")
            #     print("---------------------------")
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            ring_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            dist_ring_finger = math.sqrt((ring_finger_tip.x - ring_finger_mcp.x)**2 + (ring_finger_tip.y - ring_finger_mcp.y)**2)
            # print("distancia entre ring fingers es: " + str(dist_ring_finger))
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            dist_thumb_and_index = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
            # print("distancia entre thumb e index: " + str(dist_thumb_and_index))
            
            # if dist_thumb_and_index < 0.1:
            #     print("se juntó el pulgar con el indice---")
            pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
            dist_pinky_pip_and_ring_tip = math.sqrt((ring_finger_tip.x - pinky_pip.x)**2 + (ring_finger_tip.y - pinky_pip.y)**2)
            # print("distancia entre pynky_pip y ring_ringer_tip: " + str(dist_pinky_pip_and_ring_tip))
            
            # if dist_pinky_pip_and_ring_tip < 0.1:
            #     print("---se juntó el anular con el meñique " )
                
            ring_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            dist_ring_finger_pip_and_middle_tip = math.sqrt((ring_finger_pip.x - middle_finger_tip.x)**2 + (ring_finger_pip.y - middle_finger_tip.y)**2)
            # print("distancia entre ring_finger_pip y middle_finger_tip: " + str(dist_ring_finger_pip_and_middle_tip))
            # if dist_ring_finger_pip_and_middle_tip < 0.1:
            #     print("se juntó el mediano con el anular * * *" )
            if dist_thumb_and_index < 0.1 and dist_pinky_pip_and_ring_tip < 0.1 and dist_ring_finger_pip_and_middle_tip < 0.1:
               # print()
                log_event("---- SE ESTÁ ESCRIBIENDO ----" + str(datetime.now()))
            # c) Podrías detectar un puño cerrado si todos los tips (Thumb Tip, Index Tip, etc.)
            #    están por debajo de sus MCP, o si la distancia entre ellos y la palma es mínima, etc.
            #
            # Por ejemplo (muy simplificado), para ver si todos los dedos están por debajo
            # de un cierto valor (simbolizando un puño):
            finger_tips_indices = [4, 8, 12, 16, 20]  # pulgar, índice, medio, anular, meñique
            # Convertimos a un array:
            finger_tips_positions = [hand_landmarks.landmark[i] for i in finger_tips_indices]
            # Chequeo sencillo: si todos los tips están por debajo (y mayor que) la muñeca
            # (es decir más cerca de la parte inferior de la imagen, = puño cerrado).
            # if all(tip.y > wrist_y for tip in finger_tips_positions):
            #     print("Puño cerrado")
            # NOTA: Para lógica de gestos más robusta, deberás revisar distancias o ángulos
            # entre articulaciones, pero esto te da la idea general.
    # Para pose
    # DONDE SE USA CODO, HOMBRO Y MUÑECA
    if results_pose.pose_landmarks:
        # dibujar o analizar hombros/codos
        mp_drawing.draw_landmarks(
            frame,
            results_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
        landmarks_pose = results_pose.pose_landmarks.landmark
        #  - Right Wrist (16)
        #  - Left Elbow (13)
        #  - Left Wrist (15)
        #  - Right Elbow (14)
        right_wrist = landmarks_pose[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_elbow = landmarks_pose[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = landmarks_pose[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_elbow = landmarks_pose[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        nose = landmarks_pose[mp_pose.PoseLandmark.NOSE.value]
        if (right_elbow.y < nose.y) and (right_wrist.y < right_elbow.y):
            #print("------BRAZO DERECHO LEVANTADO-------" + str(datetime.now()))
            log_event("------BRAZO DERECHO LEVANTADO-------" + str(datetime.now()))
        if (left_elbow.y < nose.y) and (left_wrist.y < left_elbow.y):
            #print("------BRAZO IZQUIERDO LEVANTADO-----" + str(datetime.now()))
            log_event("------BRAZO IZQUIERDO LEVANTADO-----" + str(datetime.now()))
                        # Calculamos distancias Euclidianas en 2D (x,y)
        # 1) Distancia: muñeca derecha vs. codo izquierdo
        dist_rw_le = math.sqrt(
            (right_wrist.x - left_elbow.x)**2 +
            (right_wrist.y - left_elbow.y)**2
        )
        # print("distancia entre muñeca derecha y codo izquierdo" + str(dist_rw_le))
        # 2) Distancia: muñeca izquierda vs. codo derecho
        dist_lw_re = math.sqrt(
            (left_wrist.x - right_elbow.x)**2 +
            (left_wrist.y - right_elbow.y)**2
        )
        # print("distancia entre muñeca ixquierda y codo derecho" + str(dist_lw_re))
        # Definimos un umbral en coordenadas normalizadas
        threshold = 0.2  # Ajusta según lo que consideres “cerca”
        # if dist_rw_le < threshold:
        #     print("Muñeca derecha CERCA del codo izquierdo!")
        # if dist_lw_re < threshold:
        #     print("Muñeca izquierda CERCA del codo derecho!")
        if dist_rw_le < threshold and dist_lw_re < threshold:
            #print("--- SE ESTÁ CRUZANDO DE BRAZOS ---" + str(datetime.now()))
            log_event("--- SE ESTÁ CRUZANDO DE BRAZOS ---" + str(datetime.now()))
     # -------------------------
    #  LÓGICA DE FACE MESH
    # -------------------------
    #ESCLUSIVAMENTE DE LA CARA
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            # (Opcional) Dibuja la malla de la cara
            # mp_drawing.draw_landmarks(
            #     frame,
            #     face_landmarks,
            #     mp_face_mesh.FACEMESH_CONTOURS
            # )
            # Extraemos puntos para los párpados
            # Ojo izquierdo
            left_eye_top = face_landmarks.landmark[159]
            left_eye_bottom = face_landmarks.landmark[145]
            dist_left_eye = math.sqrt(
                (left_eye_top.x - left_eye_bottom.x)**2 +
                (left_eye_top.y - left_eye_bottom.y)**2
            )
            # print("distancia ojo izquierdo" + str(dist_left_eye))
            # Ojo derecho
            right_eye_top = face_landmarks.landmark[386]
            right_eye_bottom = face_landmarks.landmark[374]
            dist_right_eye = math.sqrt(
                (right_eye_top.x - right_eye_bottom.x)**2 +
                (right_eye_top.y - right_eye_bottom.y)**2
            )
            # Umbral para considerar "ojo cerrado"
            threshold = 0.01
            # Verificamos cada ojo
            ojo_izq_cerrado = dist_left_eye < threshold
            ojo_der_cerrado = dist_right_eye < threshold
            if ojo_izq_cerrado and ojo_der_cerrado:
                #print("Ambos ojos cerrados   " + str(datetime.now()))
                log_event("Ambos ojos cerrados   " + str(datetime.now()))
            elif ojo_izq_cerrado:
                #print("Ojo izquierdo cerrado  " + str(datetime.now()))
                log_event("Ojo izquierdo cerrado  " + str(datetime.now()))
            elif ojo_der_cerrado:
                #print("Ojo derecho cerrado  " + str(datetime.now()))
                log_event("Ojo derecho cerrado  " + str(datetime.now()))
            else:
                # Opcional: no imprimir nada o imprimir "ojos abiertos"
                pass
    # LÓGICA PARA DETECTAR MANO EN LA BARBILLA
    if results_face.multi_face_landmarks and results.multi_hand_landmarks:
        # Normalmente solo tenemos 1 cara (max_num_faces=1),
        # pero por seguridad iteramos. (o tomamos la primera cara directamente)
        for face_landmarks in results_face.multi_face_landmarks:
            # Landmark de la barbilla (en Face Mesh, índice 152 suele estar cerca del mentón)
            chin_landmark = face_landmarks.landmark[152]
            chin_x, chin_y, chin_z = chin_landmark.x, chin_landmark.y, chin_landmark.z
            # Podrías iterar sobre las manos detectadas
            for hand_landmarks in results.multi_hand_landmarks:
                # Ejemplo: usar la muñeca para aproximar la posición de la mano
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z
                # Calcular la distancia Euclidiana en 3D:
                dist_chin_hand = math.sqrt(
                    (chin_x - wrist_x)**2 + 
                    (chin_y - wrist_y)**2 + 
                    (chin_z - wrist_z)**2
                )
                # print("la distancia entre la muñeca y la barbilla es: " + str(dist_chin_hand))
                # Escoge un umbral en coordenadas normalizadas
                # (a veces ~0.05-0.1 puede servir, depende de la cámara/distancia)
                threshold = 0.16
                if dist_chin_hand < threshold:
                    #print("------- MANO SOBRE LA BARBILLA ---------" + str(datetime.now()))
                    log_event("------- MANO SOBRE LA BARBILLA ---------" + str(datetime.now()))
    # Codifica el fotograma como JPEG y envía la secuencia de bytes
    ret, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    
    last_frame = frame          # para un /video_feed opcional
    return ('', 204)            # sin contenido




print("Puerto en uso:", port)


if __name__ == '__main__':
    #indices = listar_camaras_disponibles()
    #print("Cámaras detectadas:", indices)
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)

    
