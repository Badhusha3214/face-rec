import React, { useEffect, useRef, useState } from 'react';
import * as faceapi from 'face-api.js';
import './FaceRecognition.css';



function FaceRecognition() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [error, setError] = useState(null);
  const [labeledFaceDescriptors, setLabeledFaceDescriptors] = useState(null);
  const [lastSpokenTimes, setLastSpokenTimes] = useState({});

  const loadLabeledImages = async () => {
    try {
      // You can start with just one person for testing
      const labels = ['badhusha','abhishek'];
      const descriptions = [];

      for (const label of labels) {
        console.log(`Loading images for ${label}...`);
        const descriptors = [];
        
        for (let i = 1; i <= 2; i++) {
          try {
            // First, check if the image exists
            const response = await fetch(`/labeled_images/${label}/${i}.jpg`);
            if (!response.ok) {
              console.warn(`Image ${i} for ${label} not found, skipping...`);
              continue;
            }

            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);
            const img = await faceapi.fetchImage(imageUrl);
            
            console.log(`Processing image ${i} for ${label}...`);
            const detection = await faceapi
              .detectSingleFace(img)
              .withFaceLandmarks()
              .withFaceDescriptor();
              
            if (detection) {
              descriptors.push(detection.descriptor);
              console.log(`✅ Successfully processed image ${i} for ${label}`);
            } else {
              console.warn(`No face detected in image ${i} for ${label}`);
            }
            
            // Clean up the object URL
            URL.revokeObjectURL(imageUrl);
            
          } catch (imgError) {
            console.error(`Error processing image ${i} for ${label}:`, imgError);
          }
        }

        if (descriptors.length > 0) {
          descriptions.push(new faceapi.LabeledFaceDescriptors(label, descriptors));
          console.log(`✅ Successfully loaded ${descriptors.length} images for ${label}`);
        } else {
          console.warn(`⚠️ No valid face descriptors found for ${label}`);
        }
      }

      if (descriptions.length > 0) {
        setLabeledFaceDescriptors(descriptions);
        console.log('✅ Face recognition data loaded successfully');
      } else {
        throw new Error('No valid face descriptors were created');
      }
      
    } catch (error) {
      console.error('❌ Error loading labeled images:', error);
      setError('Failed to load face recognition data');
    }
  };

  useEffect(() => {
    const loadModels = async () => {
        try {
          await faceapi.nets.ssdMobilenetv1.loadFromUri('/models');
          console.log('ssdMobilenetv1 loaded successfully');
      
          await faceapi.nets.faceLandmark68Net.loadFromUri('/models'); // Using faceLandmark68Net to specify the correct net
          console.log('faceLandmark68Net loaded successfully');
      
          await faceapi.nets.faceRecognitionNet.loadFromUri('/models');
          console.log('faceRecognitionNet loaded successfully');
      
          await faceapi.nets.faceExpressionNet.loadFromUri('/models');
          console.log('faceExpressionNet loaded successfully');
      
          await loadLabeledImages();
      
          setModelsLoaded(true);
          startVideo();
        } catch (err) {
          console.error('Error loading models:', err);
          setError('Failed to load face recognition models');
        }
      };
      
       

    loadModels();
  }, []);

  const startVideo = () => {
    if (videoRef.current) {
      navigator.mediaDevices.getUserMedia({ video: {} })
        .then((stream) => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }
        })
        .catch((err) => {
          console.error('Error accessing webcam:', err);
          setError('Unable to access webcam');
        });
    }
  };

  const handleVideoPlay = async () => {
    if (!modelsLoaded) return; // Prevents running if models aren't loaded
  
    const video = videoRef.current;
    const canvas = canvasRef.current;
  
    if (!video || !canvas) return;
  
    const displaySize = { width: video.width, height: video.height };
    faceapi.matchDimensions(canvas, displaySize);
  
    const detectFaces = async () => {
      if (modelsLoaded && labeledFaceDescriptors) {
        const detections = await faceapi
          .detectAllFaces(video, new faceapi.SsdMobilenetv1Options())
          .withFaceLandmarks()
          .withFaceDescriptors()
          .withFaceExpressions();
  
        const resizedDetections = faceapi.resizeResults(detections, displaySize);
  
        const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
  
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
  
        resizedDetections.forEach(detection => {
          const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
          const box = detection.detection.box;
          const drawBox = new faceapi.draw.DrawBox(box, { 
            label: bestMatch.toString(),
            lineWidth: 2 
          });
          drawBox.draw(canvas);

          if (bestMatch.label !== 'unknown' && bestMatch.distance < 0.6) {
            speakName(bestMatch.label);
          }
        });
  
        faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
        faceapi.draw.drawFaceExpressions(canvas, resizedDetections);
      }
      requestAnimationFrame(detectFaces);
    };
  
    detectFaces();
  };
    

  const speakName = (name) => {
    if ('speechSynthesis' in window) {
      const now = Date.now();
      const lastSpokenTime = lastSpokenTimes[name] || 0;
      const timeElapsed = now - lastSpokenTime;
      const ONE_MINUTE = 60000; // 60000 milliseconds = 1 minute

      // Only speak if we haven't spoken to this person in the last minute
      if (timeElapsed > ONE_MINUTE) {
        const utterance = new SpeechSynthesisUtterance();
        
        // Special greeting for Badhusha
        if (name.toLowerCase() === 'badhusha') {
          utterance.text = "Hi sir, how are you?";
        } else if (name.toLowerCase() === 'abhishek') {
          utterance.text = "Hi Abhi, how are you? How can I help you today?";
        } else {
          utterance.text = `Hello ${name}, welcome!`;
        }

        // Customize the speech
        utterance.volume = 1;
        utterance.rate = 1;
        utterance.pitch = 1;
        
        // Choose an English voice if available
        const voices = window.speechSynthesis.getVoices();
        const englishVoice = voices.find(voice => voice.lang === 'en-US');
        if (englishVoice) {
          utterance.voice = englishVoice;
        }

        // Optional: Add events for logging
        utterance.onstart = () => {
          console.log(`Started speaking to ${name}`);
        };
        
        utterance.onend = () => {
          console.log(`Finished speaking to ${name}`);
        };

        window.speechSynthesis.speak(utterance);
        
        // Update the last spoken time for this specific person
        setLastSpokenTimes(prev => ({
          ...prev,
          [name]: now
        }));
      }
    }
  };

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  return (
    <div className="face-recognition-container">
      <h1>Face Recognition</h1>
      <div className="video-container">
        <video
          ref={videoRef}
          autoPlay
          muted
          width={640}
          height={480}
          onPlay={handleVideoPlay}
          className="video"
        />
        <canvas
          ref={canvasRef}
          className="canvas"
          width={640}
          height={480}
        />
      </div>
      <div className="status">
        {modelsLoaded ? 'Models Loaded. Detecting Faces...' : 'Loading models...'}
      </div>
    </div>
  );
}

export default FaceRecognition;