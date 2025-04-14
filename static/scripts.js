 // Seleccionamos el video y el canvas
 const videoElement = document.querySelector('.input_video');
 const canvasElement = document.querySelector('.output_canvas');
 const canvasCtx = canvasElement.getContext('2d');

 // 1) Configurar la solución de Hands en JavaScript
 const hands = new Hands({
   locateFile: (file) => {
     // Indica dónde buscar los archivos del modelo
     return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
   }
 });
 hands.setOptions({
   maxNumHands: 2,            // Número máximo de manos a detectar
   modelComplexity: 1,        // Complejidad del modelo (0,1)
   minDetectionConfidence: 0.5,
   minTrackingConfidence: 0.5
 });

 // 2) Función que se llamará cuando tengamos resultados
 hands.onResults((results) => {
   // Limpiamos el canvas
   canvasCtx.save();
   canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
   
   // Dibujamos la imagen de la cámara en el canvas
   canvasCtx.drawImage(
     results.image, 0, 0, canvasElement.width, canvasElement.height
   );

   // Verificamos si se detectaron manos
   if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
     for (const landmarks of results.multiHandLandmarks) {
       // Dibuja las conexiones y puntos de la mano (opcional)
       drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
         color: '#00FF00', // color de las líneas
         lineWidth: 2
       });
       drawLandmarks(canvasCtx, landmarks, {
         color: '#FF0000', // color de los puntos
         lineWidth: 1
       });

       // 3) Calcular y dibujar un bounding box alrededor de la mano
       const xArray = landmarks.map((lm) => lm.x);
       const yArray = landmarks.map((lm) => lm.y);

       // MediaPipe da coordenadas normalizadas [0..1], escalamos al tamaño del canvas
       const xMin = Math.min(...xArray) * canvasElement.width;
       const xMax = Math.max(...xArray) * canvasElement.width;
       const yMin = Math.min(...yArray) * canvasElement.height;
       const yMax = Math.max(...yArray) * canvasElement.height;

       canvasCtx.strokeStyle = 'blue';
       canvasCtx.lineWidth = 2;
       canvasCtx.strokeRect(xMin, yMin, xMax - xMin, yMax - yMin);
     }
   }
   canvasCtx.restore();
 });

 // 4) Acceder a la cámara del usuario
 async function startCamera() {
   // Pedir permisos de cámara
   const stream = await navigator.mediaDevices.getUserMedia({ video: true });
   videoElement.srcObject = stream;
   videoElement.play();

   // Ajustar el tamaño del canvas si deseas
   canvasElement.width = 640;
   canvasElement.height = 480;
 }

 // Iniciamos la cámara
 startCamera();

 // 5) Bucle de animación: enviar cada frame a MediaPipe
 function onFrame() {
   hands.send({ image: videoElement });
   requestAnimationFrame(onFrame);
 }
 onFrame();