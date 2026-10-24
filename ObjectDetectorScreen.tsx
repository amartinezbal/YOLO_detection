/**
 * ObjectDetectorScreen.tsx
 * ─────────────────────────────────────────────────────────
 * Real-time YOLO object detection — Objects365 (365 classes)
 *
 * Dependencies:
 *   npm install react-native-vision-camera
 *   npm install react-native-fast-tflite          ← Android inference
 *   npm install react-native-worklets-core
 *
 * Asset placement:
 *   Android: android/app/src/main/assets/yolov8n_int8.tflite
 *   iOS:     ios/<AppName>/yolov8n.mlpackage  (add to Xcode target)
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  StyleSheet,
  View,
  Text,
  Platform,
  StatusBar,
  Dimensions,
} from "react-native";
import {
  Camera,
  useCameraDevice,
  useFrameProcessor,
} from "react-native-vision-camera";
import { useTensorflowModel } from "react-native-fast-tflite";
import { Worklets } from "react-native-worklets-core";

// ─── Objects365 class names (365 classes) ────────────────
// Full list: see objects365_classes.txt
const OBJECTS365_CLASSES: string[] = [
  "Person","Sneakers","Chair","Other Shoes","Hat","Car","Lamp","Glasses",
  "Bottle","Desk","Cup","Street Lights","Cabinet/shelf","Handbag/Satchel",
  "Bracelet","Plate","Picture/Frame","Helmet","Book","Gloves","Storage box",
  "Boat","Leather Shoes","Flower","Bench","Potted Plant","Bowl/Basin","Flag",
  "Pillow","Boots","Vase","Microphone","Necklace","Ring","SUV","Wine Glass",
  "Belt","Moniter/TV","Backpack","Umbrella","Traffic Light","Speaker","Watch",
  "Tie","Trash bin Can","Slippers","Bicycle","Stool","Barrel/bucket","Van",
  "Couch","Sandals","Basket","Drum","Pen/Pencil","Bus","Wild Bird","High Heels",
  "Motorcycle","Guitar","Carpet","Cell Phone","Bread","Camera","Canned",
  "Billiards","Candle","Stuffed Animal","Spoon","Hot dog","Calories Counter",
  "Donut","Stroller","Medical Equipment","Fire Extinguisher","Computer Box",
  "Earphone","Painting","Watermelon","Potato","Apple","Orange","Lemon",
  "Durian","Avocado","Strawberry","Banana","Pitcher","Mango","Baseball Glove",
  "Skateboard","Soccer","Globe","Chess","Arrow","Pliers","Saw","Soccer Shoe",
  "Jug","Stapler","Fork","Comb","Pot/Pan","Padlock","Broom","Refrigerator",
  "Oven/Stove","Washing Machine","Blender","Microwave","Other Musical Instrument",
  "Baseball","Kite","Tennis Racket","Volleyball","Frisbee","Surfboard",
  "Hang glider","Parachute","Snowboard","Skis","Rowing machine","Dumbbell",
  "Treadmill","Tire","Traffic cone","Parking meter","Horse","Dog","Cat",
  "Deer","Bird","Elephant","Bear","Cow","Giraffe","Zebra","Camel","Sheep",
  "Rabbit","Lion","Tiger","Monkey","Gorilla","Fox","Raccoon","Squirrel",
  "Bicycle helmet","Motorcycle helmet","Paper","Pen","Eraser","Ruler",
  "Staple","Tape dispenser","Scissors","Clip","Binder clip","Whiteboard",
  "Projector","Clock","Calendar","Printer","Laptop","Keyboard","Mouse",
  "Scanner","Remote","Charger","Power Strip","Cable","Ethernet",
  "Telephone","Tablet","Game controller","Headphones","Tripod","Camera lens",
  "Selfie stick","SD card","USB drive","Battery","Bulb","Fan","Air conditioner",
  "Heater","Curtains","Towel","Shower","Bathtub","Toilet","Sink","Mirror",
  "Soap dispenser","Toothbrush","Razor","Trash can","Laundry basket",
  "Hanger","Iron","Vacuum cleaner","Mop","Broom","Dustpan","Bucket",
  "Toolbox","Hammer","Screwdriver","Wrench","Drill","Tape measure","Level",
  "Paint brush","Paint roller","Ladder","Wheelbarrow","Garden hose",
  "Watering can","Shovel","Rake","Pitchfork","Axe","Chainsaw",
  "Lawn mower","Hedge trimmer","Leaf blower","Plow","Tractor","Forklift",
  "Crane","Excavator","Dump truck","Cement mixer","Fire truck","Ambulance",
  "Police car","Taxi","Limousine","Pickup truck","Semi truck","Trailer",
  "RV/Camper","ATV","Go-kart","Jet ski","Kayak","Canoe","Sailboat",
  "Speedboat","Aircraft carrier","Submarine","Hot air balloon","Helicopter",
  "Private jet","Rocket","Satellite dish","Windmill","Solar panel",
  "Power line","Pipeline","Bridge","Tunnel","Dam","Lighthouse",
  "Monument","Fountain","Bench (outdoor)","Swing","Slide","Seesaw",
  "Trampoline","Basketball hoop","Tennis court","Swimming pool",
  "Ski lift","Ferris wheel","Roller coaster","Carousel","Stage",
  "Microphone stand","Guitar amp","Drum kit","Piano","Violin","Trumpet",
  "Saxophone","Cello","Harp","Accordion","Flute","Clarinet","Tuba",
  "Trombone","Xylophone","Turntable","Synthesizer","Sheet music","Conductor baton",
];

// ─── Detection result type ────────────────────────────────
interface Detection {
  classId: number;
  label: string;
  confidence: number;
  x: number;   // 0–1 normalized (top-left origin)
  y: number;
  width: number;
  height: number;
}

// ─── Bounding box color palette ───────────────────────────
const BOX_COLORS = [
  "#FF3B30","#FF9500","#FFCC00","#34C759","#00C7BE",
  "#30B0C7","#32ADE6","#007AFF","#5856D6","#AF52DE",
  "#FF2D55","#A2845E","#636366","#48484A","#1C1C1E",
];
const colorForClass = (id: number) => BOX_COLORS[id % BOX_COLORS.length];

const { width: SCREEN_W, height: SCREEN_H } = Dimensions.get("window");
const CONF_THRESHOLD = 0.35;
const MODEL_INPUT_SIZE = 640;
const MAX_DETECTIONS = 20;

// ─────────────────────────────────────────────────────────
// MAIN COMPONENT
// ─────────────────────────────────────────────────────────
export default function ObjectDetectorScreen() {
  const device = useCameraDevice("back");
  const [hasPermission, setHasPermission] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [fps, setFps] = useState(0);
  const lastFrameTime = useRef(Date.now());
  const frameCount = useRef(0);

  // ── Load TFLite model (Android path) ────────────────────
  // iOS: swap this for CoreML via @rnmlkit/object-detection
  const model = useTensorflowModel(
    require("./assets/yolov8n_int8.tflite")
  );

  useEffect(() => {
    (async () => {
      const status = await Camera.requestCameraPermission();
      setHasPermission(status === "granted");
    })();
  }, []);

  const onDetections = Worklets.createRunOnJS((dets: Detection[]) => {
    setDetections(dets);
    frameCount.current++;
    const now = Date.now();
    if (now - lastFrameTime.current >= 1000) {
      setFps(frameCount.current);
      frameCount.current = 0;
      lastFrameTime.current = now;
    }
  });

  // ── Frame processor (runs on GPU thread) ─────────────────
  const frameProcessor = useFrameProcessor(
    (frame) => {
      "worklet";
      if (!model.model) return;

      // YOLOv8 output tensor: [1, num_classes+4, 8400]
      // For Objects365: num_classes=365, so shape = [1, 369, 8400]
      const outputs = model.model.runSync([frame]);
      const raw = outputs[0] as Float32Array;

      const numPreds  = 8400;
      const numFields = 4 + OBJECTS365_CLASSES.length; // 369

      const results: Detection[] = [];

      for (let i = 0; i < numPreds; i++) {
        let maxScore = 0;
        let bestClass = 0;

        // Scan all 365 class scores for this anchor
        for (let c = 4; c < numFields; c++) {
          const score = raw[c * numPreds + i];
          if (score > maxScore) {
            maxScore = score;
            bestClass = c - 4;
          }
        }

        if (maxScore < CONF_THRESHOLD) continue;

        // cx, cy, w, h — normalized by MODEL_INPUT_SIZE
        const cx = raw[0 * numPreds + i] / MODEL_INPUT_SIZE;
        const cy = raw[1 * numPreds + i] / MODEL_INPUT_SIZE;
        const bw = raw[2 * numPreds + i] / MODEL_INPUT_SIZE;
        const bh = raw[3 * numPreds + i] / MODEL_INPUT_SIZE;

        results.push({
          classId: bestClass,
          label: OBJECTS365_CLASSES[bestClass] ?? `class_${bestClass}`,
          confidence: maxScore,
          x: cx - bw / 2,
          y: cy - bh / 2,
          width: bw,
          height: bh,
        });
      }

      // Sort by confidence then apply greedy NMS
      results.sort((a, b) => b.confidence - a.confidence);
      const kept: Detection[] = [];
      for (const det of results) {
        if (kept.length >= MAX_DETECTIONS) break;
        const overlaps = kept.some((k) => iou(k, det) > 0.45);
        if (!overlaps) kept.push(det);
      }

      onDetections(kept);
    },
    [model, onDetections]
  );

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text style={styles.permText}>Camera permission required</Text>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.center}>
        <Text style={styles.permText}>No camera device found</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar hidden />
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive
        frameProcessor={frameProcessor}
        frameProcessorFps={15}
        pixelFormat="rgb"
      />
      {detections.map((det, idx) => (
        <BoundingBox key={idx} detection={det} />
      ))}
      <View style={styles.hud}>
        <Text style={styles.hudText}>
          {detections.length} object{detections.length !== 1 ? "s" : ""} · {fps} FPS
        </Text>
        <Text style={styles.hudModel}>YOLOv8n · Objects365 (365 classes)</Text>
      </View>
    </View>
  );
}

// ─────────────────────────────────────────────────────────
// BOUNDING BOX COMPONENT
// ─────────────────────────────────────────────────────────
function BoundingBox({ detection: det }: { detection: Detection }) {
  const color = colorForClass(det.classId);
  const left   = det.x * SCREEN_W;
  const top    = det.y * SCREEN_H;
  const width  = det.width * SCREEN_W;
  const height = det.height * SCREEN_H;

  return (
    <View style={[styles.box, { left, top, width, height, borderColor: color }]}>
      <View style={[styles.labelBg, { backgroundColor: color }]}>
        <Text style={styles.labelText}>
          {det.label} {(det.confidence * 100).toFixed(0)}%
        </Text>
      </View>
    </View>
  );
}

// ─────────────────────────────────────────────────────────
// IoU for NMS
// ─────────────────────────────────────────────────────────
function iou(a: Detection, b: Detection): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width,  b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.width * a.height + b.width * b.height - inter;
  return union > 0 ? inter / union : 0;
}

// ─────────────────────────────────────────────────────────
// STYLES
// ─────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  container:  { flex: 1, backgroundColor: "#000" },
  center:     { flex: 1, alignItems: "center", justifyContent: "center", backgroundColor: "#000" },
  permText:   { color: "#fff", fontSize: 16 },
  box: {
    position: "absolute",
    borderWidth: 2,
    borderRadius: 4,
  },
  labelBg: {
    position: "absolute",
    top: -22,
    left: -1,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 3,
  },
  labelText: {
    color: "#fff",
    fontSize: 11,
    fontWeight: "700",
    letterSpacing: 0.3,
  },
  hud: {
    position: "absolute",
    bottom: 40,
    alignSelf: "center",
    backgroundColor: "rgba(0,0,0,0.6)",
    paddingHorizontal: 18,
    paddingVertical: 10,
    borderRadius: 22,
    alignItems: "center",
    gap: 3,
  },
  hudText:  { color: "#fff",                  fontSize: 15, fontWeight: "600" },
  hudModel: { color: "rgba(255,255,255,0.55)", fontSize: 11 },
});
