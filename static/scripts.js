/* static/js/cams.js */
document.addEventListener('DOMContentLoaded', () => {
  (async () => {
    /* 0. contenedor padre */
    const camsContainer = document.getElementById('cams');
    if (!camsContainer){
      console.error('Falta el div id="cams" en el HTML');
      return;
    }

    try {
      /* 1.  Pedimos permiso (una sola cámara cualquiera)            */
      const tmp = await navigator.mediaDevices.getUserMedia({video:true});
      tmp.getTracks().forEach(t => t.stop());  // liberamos enseguida

      /* 2.  Enumeramos todos los dispositivos videoinput            */
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cams = devices.filter(d => d.kind === 'videoinput');

      console.log(`Cámaras detectadas: ${cams.length}`);

      /* 3.  Recorremos cada cámara                                  */
      for (let i = 0; i < cams.length; i++){
        const cam = cams[i];
        try{
          const stream = await navigator.mediaDevices.getUserMedia({
            video: {deviceId: {exact: cam.deviceId}},
            audio: false
          });

          /* 4.  Creamos tarjeta y reproducimos                       */
          const card  = document.createElement('div');
          card.className = 'cam';
          card.innerHTML = `
            <h3>Cámara ${i}</h3>
            <video autoplay muted playsinline></video>
          `;
          const video = card.querySelector('video');
          video.srcObject = stream;

          camsContainer.appendChild(card);
          console.log(`Cámara [${i}] añadida: ${cam.label || '(sin nombre)'}`);
          //---------------------
         /* 5.  Canvas y envío periódico a /upload                   */
      const canvas = document.createElement('canvas');
      const ctx    = canvas.getContext('2d');
      const track  = stream.getVideoTracks()[0];
      const {width, height} = track.getSettings();
      canvas.width  = width  || 640;
      canvas.height = height || 480;

      setInterval(() => {
        if (video.readyState < 2) return;      // aún no hay frame
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(blob => {
          fetch('/upload', {
            method: 'POST',
            body:   blob,
            // opcional: identifica la cámara en un header
            headers: { 'X-Cam-Id': String(i) }
          });
        }, 'image/jpeg', 0.7);
      }, 100);   // 100 ms  (10 fps) – ajusta a tu gusto

          // --------------------

        }catch(err){
          console.warn(`No se pudo abrir la cámara [${i}] – ${cam.label}:`, err);
        }
      }

    } catch(err){
      console.error('Permiso denegado o error al acceder a dispositivos:', err);
    }
  })();
});
