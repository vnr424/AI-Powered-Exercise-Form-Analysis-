from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import json
import re
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import joblib

MODEL_PATH      = SCRIPTS_DIR.parent / "models/random_forest_tuned.joblib"
SCALER_PATH     = SCRIPTS_DIR.parent / "models/feature_scaler_tuned.joblib"
POSE_MODEL_PATH = str(SCRIPTS_DIR.parent / "models/pose_landmarker_heavy.task")

print("Loading ML models...")
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print(" ML models loaded")

base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.PoseLandmarker.create_from_options(options)
print(" Pose detector ready")

from enhanced_explainability import AnatomicalExpertSystem
from rep_counter import RepCounter
from exercise_config import ExerciseConfig
from anatomical_heatmap import AnatomicalHeatmap

anatomical_expert = AnatomicalExpertSystem()
heatmap_generator = AnatomicalHeatmap(model)
print(" Medical expert system ready")
print(" Heatmap generator ready")

try:
    exercises = ExerciseConfig.get_available_exercises()
    DEFAULT_EXERCISE_KEY    = exercises[0][0]
    DEFAULT_EXERCISE_CONFIG = ExerciseConfig.get_exercise_config(DEFAULT_EXERCISE_KEY)
    print(f" Default exercise: {DEFAULT_EXERCISE_CONFIG['name']}")
except Exception as e:
    print(f"  Could not load exercise config: {e}")
    DEFAULT_EXERCISE_KEY    = "bench_press"
    DEFAULT_EXERCISE_CONFIG = {'name':'Bench Press','use_hip_features':False,'elbow_angle_range':(45,75),'required_landmarks':['left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist']}

print(" System ready! Waiting for connections...")

LANDMARKS = {'left_shoulder':11,'right_shoulder':12,'left_elbow':13,'right_elbow':14,'left_wrist':15,'right_wrist':16,'left_hip':23,'right_hip':24}

def calculate_angle(p1, p2, p3):
    v1 = np.array([p1.x-p2.x, p1.y-p2.y])
    v2 = np.array([p3.x-p2.x, p3.y-p2.y])
    cos_angle = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_angle,-1.0,1.0))))

def calculate_distance(p1, p2):
    return float(np.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2))

def extract_features(pose_landmarks, exercise_config):
    lm = pose_landmarks[0]
    ls = lm[LANDMARKS['left_shoulder']];  rs = lm[LANDMARKS['right_shoulder']]
    le = lm[LANDMARKS['left_elbow']];     re = lm[LANDMARKS['right_elbow']]
    lw = lm[LANDMARKS['left_wrist']];     rw = lm[LANDMARKS['right_wrist']]
    use_hip = exercise_config.get('use_hip_features', False)
    if use_hip:
        lh = lm[LANDMARKS['left_hip']]; rh = lm[LANDMARKS['right_hip']]
    lea = calculate_angle(ls,le,lw); rea = calculate_angle(rs,re,rw)
    avg = (lea+rea)/2
    lsa = calculate_angle(lw,ls,le) if not use_hip else calculate_angle(le,ls,lh)
    rsa = calculate_angle(rw,rs,re) if not use_hip else calculate_angle(re,rs,rh)
    wx=(lw.x+rw.x)/2; wy=(lw.y+rw.y)/2; wz=(lw.z+rw.z)/2
    bvt=abs(lw.y-rw.y); bho=abs(lw.x-rw.x); bco=abs((lw.x+rw.x)/2-0.5)
    shd=abs(ls.y-rs.y); ehd=abs(le.y-re.y); whd=abs(lw.y-rw.y)
    hd=0.0; lhy=0.0; rhy=0.0
    if use_hip: hd=abs(lh.y-rh.y); lhy=lh.y; rhy=rh.y
    cx=0.5
    shs=abs(abs(ls.x-cx)-abs(rs.x-cx)); ehs=abs(abs(le.x-cx)-abs(re.x-cx))
    ead=abs(lea-rea); sad=abs(lsa-rsa)
    if use_hip: ss=(shd*2.0+ehd*1.5+whd*1.5+hd*1.0+(ead/100.0)+(sad/100.0))/7.5
    else:       ss=(shd*2.5+ehd*2.0+whd*2.0+(ead/100.0)+(sad/100.0))/6.5
    sw=calculate_distance(ls,rs); smx=(ls.x+rs.x)/2
    if use_hip: ro=abs(smx-(lh.x+rh.x)/2)
    else:       ro=abs(smx-(le.x+re.x)/2)
    return {
        'left_elbow_angle':lea,'right_elbow_angle':rea,'avg_elbow_angle':avg,
        'wrist_x':wx,'wrist_y':wy,'wrist_z':wz,'shoulder_width':sw,'shoulder_mid_x':smx,
        'retraction_offset':ro,'wrist_y_position':wy,'elbow_angle_diff':ead,
        'elbow_symmetry':min(lea,rea)/(max(lea,rea)+1e-6),
        'wrist_x_normalized':wx/(sw+1e-6),'retraction_normalized':ro/(sw+1e-6),
        'alignment_score':abs(smx-wx),'has_flare':1 if avg>75 else 0,
        'extreme_flare':1 if avg>120 else 0,'avg_elbow_angle_squared':avg**2,
        'left_wrist_x':lw.x,'left_wrist_y':lw.y,'right_wrist_x':rw.x,'right_wrist_y':rw.y,
        'bar_vertical_tilt':bvt,'bar_horizontal_offset':bho,'bar_center_offset':bco,
        'left_shoulder_x':ls.x,'left_shoulder_y':ls.y,'right_shoulder_x':rs.x,'right_shoulder_y':rs.y,
        'left_elbow_x':le.x,'left_elbow_y':le.y,'right_elbow_x':re.x,'right_elbow_y':re.y,
        'left_shoulder_angle':lsa,'right_shoulder_angle':rsa,
        'left_hip_y':lhy,'right_hip_y':rhy,
        'shoulder_height_diff':shd,'elbow_height_diff':ehd,'wrist_height_diff':whd,'hip_height_diff':hd,
        'shoulder_horizontal_symmetry':shs,'elbow_horizontal_symmetry':ehs,'shoulder_angle_diff':sad,
        'symmetry_score':ss,
        'has_shoulder_asymmetry':1 if (shd>0.05 or sad>15) else 0,
        'has_elbow_asymmetry':1 if (ehd>0.05 or ead>15) else 0,
        'has_wrist_asymmetry':1 if whd>0.05 else 0,
        'has_hip_tilt':1 if (hd>0.05 and use_hip) else 0,
        'has_any_asymmetry':1 if (shd>0.05 or sad>15 or ehd>0.05 or ead>15 or whd>0.05) else 0,
        'overall_symmetry_grade':'A' if ss<0.05 else('B' if ss<0.10 else('C' if ss<0.15 else 'D')),
        'exercise_type':exercise_config.get('name','Unknown'),
        'has_hip_data':use_hip,
    }

def predict_technique(features_dict):
    mf = {k:features_dict[k] for k in ['left_elbow_angle','right_elbow_angle','avg_elbow_angle','wrist_x','wrist_y','wrist_z','shoulder_width','shoulder_mid_x','retraction_offset','wrist_y_position','elbow_angle_diff','elbow_symmetry','wrist_x_normalized','retraction_normalized','alignment_score','has_flare','extreme_flare','avg_elbow_angle_squared']}
    df  = pd.DataFrame([mf])
    sdf = scaler.transform(df)
    pred = model.predict(sdf)[0]
    prob = model.predict_proba(sdf)[0]
    return int(pred), float(prob[pred])

def get_feedback_message(features, prediction, errors):
    avg = features.get('avg_elbow_angle',0)
    if prediction==1:
        if avg>140: return "Great form! Arms fully extended."
        elif avg<80: return "Good depth — push back up with control."
        else: return "Good technique — keep it up!"
    if errors: return f"{errors[0].error_type} — {errors[0].severity} risk."
    if avg>120: return " Elbow flare detected — tuck elbows to ~45°"
    if features.get('symmetry_score',0)>0.15: return " Asymmetric movement — check balance"
    return "Technique needs improvement — focus on form"

def get_form_quality(prediction, confidence, features):
    if prediction==1:
        if confidence>0.85: return "excellent"
        elif confidence>0.70: return "good"
        else: return "fair"
    return "poor" if features.get('avg_elbow_angle',90)>120 else "needs_improvement"

def extract_landmarks_for_ui(pose_landmarks, frame_shape):
    h,w = frame_shape[:2]
    lm  = pose_landmarks[0]
    return [{"index":idx,"x":round(lm[idx].x*w),"y":round(lm[idx].y*h),"visibility":round(lm[idx].visibility,3)} for idx in [11,12,13,14,15,16,23,24]]

def generate_heatmap_b64(frame, pose_landmarks, prediction, features, errors):
    try:
        print(f" Generating 3-panel heatmap for Grade F rep...")
        heatmap_img = heatmap_generator.generate_report_visualization(frame.copy(), pose_landmarks, prediction, features, errors)
        _, buf = cv2.imencode(".jpg", heatmap_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    except Exception as e:
        print(f" Heatmap error: {e}")
        return None

class SessionState:
    def __init__(self):
        self.rep_counter     = RepCounter(history_size=10)
        self.exercise_config = DEFAULT_EXERCISE_CONFIG
        self.frame_count     = 0
        self.last_grade      = "—"

@app.get("/")
async def root():
    return {"status":"Exercise Analysis WebSocket Server","endpoint":"/ws"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(" Client connected!")
    session = SessionState()
    try:
        while True:
            data       = await websocket.receive_text()
            frame_data = json.loads(data)
            if frame_data.get("type")=="set_exercise":
                try:
                    session.exercise_config = ExerciseConfig.get_exercise_config(frame_data.get("exercise","bench_press"))
                    session.rep_counter     = RepCounter(history_size=10)
                    await websocket.send_json({"type":"exercise_set","exercise":session.exercise_config['name']})
                except: pass
                continue
            img_bytes = base64.b64decode(frame_data['frame'].split(',')[1])
            frame     = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue
            session.frame_count += 1
            mp_image         = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = detector.detect(mp_image)
            if not detection_result.pose_landmarks:
                await websocket.send_json({"status":"no_person","grade":session.last_grade,"rep_count":session.rep_counter.rep_count,"feedback":"No person detected — adjust camera","elbow_angle":0,"form_quality":"unknown","phase":"UNKNOWN","landmarks":[],"symmetry_grade":"—","symmetry_score":0})
                continue
            try:
                features               = extract_features(detection_result.pose_landmarks, session.exercise_config)
                prediction, confidence = predict_technique(features)
                errors                 = anatomical_expert.analyze_technique(features, session.exercise_config)
                heatmap_b64            = None
                detailed_report        = ""
                rep_completed, rep_analysis = session.rep_counter.update(features, prediction, confidence, errors)
                if rep_completed:
                    session.last_grade = rep_analysis.get_quality_grade()
                    print(f" REP #{rep_analysis.rep_number} — Grade: {session.last_grade}")
                    if session.last_grade == "F":
                        heatmap_b64 = generate_heatmap_b64(frame, detection_result.pose_landmarks, prediction, features, errors)
                    detailed_report = anatomical_expert.generate_detailed_report(errors) if errors else ""

                error_list   = [{"type":e.error_type,"severity":e.severity,"muscles":[m.name for m in e.affected_muscles[:2]]} for e in errors[:3]]
                landmarks_ui = extract_landmarks_for_ui(detection_result.pose_landmarks, frame.shape)
                await websocket.send_json({
                    "status":"analyzed",
                    "grade":session.last_grade if session.rep_counter.rep_count>0 else "—",
                    "rep_count":session.rep_counter.rep_count,
                    "feedback":get_feedback_message(features,prediction,errors),
                    "elbow_angle":round(features['avg_elbow_angle'],1),
                    "left_elbow":round(features['left_elbow_angle'],1),
                    "right_elbow":round(features['right_elbow_angle'],1),
                    "form_quality":get_form_quality(prediction,confidence,features),
                    "prediction":prediction,
                    "confidence":round(confidence*100,1),
                    "phase":session.rep_counter.current_phase,
                    "symmetry_grade":features['overall_symmetry_grade'],
                    "symmetry_score":round(features['symmetry_score'],3),
                    "bar_level":"LEVEL" if features['bar_vertical_tilt']<0.05 else "UNEVEN",
                    "errors":error_list,
                    "landmarks":landmarks_ui,
                    "exercise":session.exercise_config.get('name','Bench Press'),
                    "rep_just_completed":rep_completed,
                    "frame_number":session.frame_count,
                    "heatmap":heatmap_b64,
                    "detailed_report":detailed_report if rep_completed else "",
                })
                if session.frame_count%30==0:
                    print(f" Frame {session.frame_count} | Reps:{session.rep_counter.rep_count} | Elbow:{features['avg_elbow_angle']:.1f}° | {'' if prediction==1 else ''} ({confidence*100:.0f}%)")
            except Exception as e:
                print(f" Error: {e}")
                await websocket.send_json({"status":"error","grade":session.last_grade,"rep_count":session.rep_counter.rep_count,"feedback":"Analysis error — ensure full body is visible","elbow_angle":0,"form_quality":"unknown","phase":session.rep_counter.current_phase,"landmarks":[]})
    except Exception as e:
        print(f" Connection error: {e}")
    finally:
        print(f" Client disconnected ({session.frame_count} frames, {session.rep_counter.rep_count} reps)")

@app.post("/report")
async def generate_report(data: dict):
    session_data = data.get("session", {})
    report_type  = data.get("type", "session")
    reps         = session_data.get("reps", [])
    summary      = session_data.get("summary", {})
    current      = session_data.get("current", {})
    exercise     = session_data.get("exercise", "Bench Press")
    timestamp    = session_data.get("timestamp", "")
    coaching_raw = session_data.get("coaching", "")

    GC = {"A":"#22c55e","B":"#3b82f6","C":"#eab308","D":"#f97316","F":"#ef4444","—":"#6b7280"}
    def gc(g): return GC.get(g,"#6b7280")

    # Convert Gemini markdown to styled HTML
    coaching_html = ""
    if coaching_raw:
        cr = coaching_raw
        cr = re.sub(r'^## (.+)$', r'<h3 class="ch3">\1</h3>', cr, flags=re.MULTILINE)
        cr = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', cr)
        cr = re.sub(r'^- (.+)$', r'<li>\1</li>', cr, flags=re.MULTILINE)
        cr = re.sub(r'^(\d+)\. (.+)$', r'<li><span class="step-num">\1</span>\2</li>', cr, flags=re.MULTILINE)
        cr = re.sub(r'(<li>.*?</li>\n?)+', lambda m: f'<ul>{m.group(0)}</ul>', cr, flags=re.DOTALL)
        cr = re.sub(r'\n{2,}', '</p><p>', cr)
        cr = f'<p>{cr}</p>'
        coaching_html = f'''
        <div class="section coaching">
          <h2> AI Coaching Summary — Powered by Gemini</h2>
          <div class="coaching-body">{cr}</div>
        </div>'''

    # Rep rows with heatmaps
    rep_rows = ""
    for rep in reps:
        grade   = rep.get("grade","—")
        tech    = " Incorrect" if rep.get("grade","—") == "F" else " Correct"
        etags   = "".join(f'<span class="etag">{e.get("type","")}</span>' for e in rep.get("errors",[]))
        heatmap = rep.get("heatmap")
        rep_rows += f"""
        <tr>
          <td>#{rep.get('rep_number','?')}</td>
          <td><span class="gbadge" style="background:{gc(grade)}">{grade}</span></td>
          <td>{tech}</td>
          <td>{rep.get('elbow_angle','—')}°</td>
          <td style="color:{gc(rep.get('symmetry_grade','—'))}">{rep.get('symmetry_grade','—')}</td>
          <td>{rep.get('phase','—')}</td>
          <td>{etags or '<span style="color:#22c55e">None</span>'}</td>
        </tr>"""
        if heatmap:
            rep_rows += f"""
        <tr class="heatmap-row">
          <td colspan="7" style="padding:0.5rem 1rem 1rem">
            <div style="background:#0f1117;border-radius:8px;padding:0.75rem;border:1px solid rgba(239,68,68,0.25)">
              <div style="font-size:0.7rem;color:#f97316;font-weight:600;margin-bottom:0.5rem;text-transform:uppercase;letter-spacing:0.05em">
                 Grad-CAM Heatmap — Rep #{rep.get('rep_number','?')} (Red = High Injury Risk)
              </div>
              <img src="{heatmap}" style="width:100%;max-height:220px;object-fit:contain;border-radius:6px" />
              <div style="font-size:0.65rem;color:#475569;margin-top:0.4rem">Blue = Safe &nbsp;|&nbsp; Yellow = Caution &nbsp;|&nbsp; Red/Orange = Problem areas</div>
            </div>
          </td>
        </tr>"""

    # Current frame section
    current_html = ""
    if current:
        pc   = "#22c55e" if current.get("prediction")==1 else "#ef4444"
        pt   = " CORRECT TECHNIQUE" if current.get("prediction")==1 else " INCORRECT TECHNIQUE"
        errs = "".join(f'<div class="ecard"><div class="etitle">{e.get("type","")}</div><div style="font-size:.8rem;color:{"#ef4444" if e.get("severity")=="CRITICAL" else "#f97316"}">Severity: {e.get("severity","")}</div><div style="font-size:.8rem;color:#64748b">Affected: {", ".join(e.get("muscles",[]))}</div></div>' for e in current.get("errors",[]))
        current_html = f"""
        <div class="section">
          <h2>Current Frame Analysis</h2>
          <div class="mgrid">
            <div class="mc"><div class="ml">Technique</div><div class="mv" style="color:{pc};font-size:1rem">{pt}</div></div>
            <div class="mc"><div class="ml">Confidence</div><div class="mv">{current.get('confidence','—')}%</div></div>
            <div class="mc"><div class="ml">Avg Elbow</div><div class="mv">{current.get('elbow_angle','—')}°</div></div>
            <div class="mc"><div class="ml">Left Elbow</div><div class="mv">{current.get('left_elbow','—')}°</div></div>
            <div class="mc"><div class="ml">Right Elbow</div><div class="mv">{current.get('right_elbow','—')}°</div></div>
            <div class="mc"><div class="ml">Symmetry</div><div class="mv" style="color:{gc(current.get('symmetry_grade','—'))}">{current.get('symmetry_grade','—')}</div></div>
            <div class="mc"><div class="ml">Bar Level</div><div class="mv">{' Level' if current.get('bar_level')=='LEVEL' else ' Uneven'}</div></div>
            <div class="mc"><div class="ml">Phase</div><div class="mv">{current.get('phase','—')}</div></div>
          </div>
          {f'<div style="margin-top:1.2rem">{errs}</div>' if errs else '<div class="noiss"> No issues detected</div>'}
        </div>"""

    # Session summary
    session_html = ""
    if summary and reps:
        total   = summary.get("total_reps",0)
        correct = summary.get("correct_reps",0)
        acc     = summary.get("accuracy",0)
        avg_el  = summary.get("avg_elbow","—")
        grades  = summary.get("grade_distribution",{})
        heatmap_count = sum(1 for r in reps if r.get("heatmap"))
        gdist = "".join(f'<div class="gdi"><span class="gbadge" style="background:{gc(g)}">{g}</span><span>{c} rep{"s" if c!=1 else ""}</span></div>' for g,c in grades.items() if c>0)
        session_html = f"""
        <div class="section">
          <h2>Session Summary</h2>
          <div class="mgrid">
            <div class="mc hi"><div class="ml">Total Reps</div><div class="mv big">{total}</div></div>
            <div class="mc hi"><div class="ml">Correct Reps</div><div class="mv big" style="color:#22c55e">{correct}</div></div>
            <div class="mc hi"><div class="ml">Accuracy</div><div class="mv big" style="color:{'#22c55e' if acc>=80 else '#f97316'}">{acc:.1f}%</div></div>
            <div class="mc"><div class="ml">Avg Elbow</div><div class="mv">{avg_el}°</div></div>
            <div class="mc"><div class="ml">Heatmaps</div><div class="mv" style="color:#f97316">{heatmap_count}</div></div>
          </div>
          <div class="gdist" style="margin-top:1rem">{gdist}</div>
          <h3 style="margin-top:1.5rem;color:#94a3b8;font-size:.8rem;text-transform:uppercase;margin-bottom:.5rem">Rep-by-Rep Breakdown</h3>
          <table class="rt">
            <thead><tr><th>Rep</th><th>Grade</th><th>Technique</th><th>Elbow°</th><th>Symmetry</th><th>Phase</th><th>Issues</th></tr></thead>
            <tbody>{rep_rows}</tbody>
          </table>
        </div>"""

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>FormAI Report</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Space Grotesk',sans-serif;background:#0f1117;color:#e2e8f0;padding:2rem}}
.wrap{{max-width:960px;margin:0 auto;background:#1a1f2e;border-radius:16px;overflow:hidden;box-shadow:0 0 60px rgba(0,0,0,.5)}}
.hdr{{background:linear-gradient(135deg,#1e3a5f,#0f2744,#1a1040);padding:2.5rem;border-bottom:1px solid rgba(255,255,255,.08)}}
.htop{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1.5rem}}
.logo{{font-size:1.8rem;font-weight:700;background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.meta{{text-align:right;font-size:.75rem;color:#64748b;font-family:'JetBrains Mono',monospace;line-height:1.6}}
.hdr h1{{font-size:1.5rem;font-weight:600;color:#f1f5f9;margin-bottom:.75rem}}
.badges{{display:flex;gap:.5rem;flex-wrap:wrap}}
.badge{{padding:.25rem .75rem;border-radius:999px;font-size:.7rem;font-weight:600;border:1px solid rgba(255,255,255,.15);background:rgba(255,255,255,.07);color:#94a3b8}}
.section{{padding:2rem 2.5rem;border-bottom:1px solid rgba(255,255,255,.06)}}
.section h2{{font-size:.85rem;font-weight:700;color:#60a5fa;margin-bottom:1.25rem;padding-bottom:.5rem;border-bottom:1px solid rgba(96,165,250,.2);text-transform:uppercase;letter-spacing:.05em}}
.mgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:.75rem}}
.mc{{background:#232936;border:1px solid rgba(255,255,255,.06);border-radius:10px;padding:1rem}}
.mc.hi{{border-color:rgba(96,165,250,.3);background:rgba(96,165,250,.07)}}
.ml{{font-size:.65rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.4rem;font-weight:600}}
.mv{{font-size:1.5rem;font-weight:700;color:#f1f5f9;font-family:'JetBrains Mono',monospace}}
.mv.big{{font-size:2rem}}
.gbadge{{display:inline-flex;align-items:center;justify-content:center;width:2rem;height:2rem;border-radius:6px;font-weight:700;font-size:.9rem;color:white}}
.gdist{{display:flex;gap:1rem;flex-wrap:wrap}}
.gdi{{display:flex;align-items:center;gap:.5rem;font-size:.9rem;color:#94a3b8}}
.ecard{{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2);border-radius:8px;padding:.875rem 1rem;margin-bottom:.75rem}}
.etitle{{font-weight:600;color:#fca5a5;margin-bottom:.25rem}}
.etag{{display:inline-block;padding:.15rem .5rem;background:rgba(239,68,68,.15);border:1px solid rgba(239,68,68,.3);border-radius:4px;font-size:.7rem;color:#fca5a5;margin-right:.25rem}}
.noiss{{padding:1rem;background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.2);border-radius:8px;color:#86efac;font-size:.9rem;margin-top:1rem}}
.rt{{width:100%;border-collapse:collapse;font-size:.82rem;margin-top:.75rem}}
.rt th{{text-align:left;padding:.6rem .75rem;background:#0f1117;color:#475569;font-size:.68rem;text-transform:uppercase;letter-spacing:.05em;font-weight:600}}
.rt td{{padding:.65rem .75rem;border-bottom:1px solid rgba(255,255,255,.04);color:#cbd5e1;vertical-align:middle}}
.heatmap-row td{{border-bottom:1px solid rgba(239,68,68,.1)!important}}
.coaching{{background:linear-gradient(135deg,rgba(96,165,250,.05),rgba(167,139,250,.05));border-left:3px solid #a78bfa}}
.coaching h2{{color:#a78bfa!important;border-color:rgba(167,139,250,.3)!important}}
.coaching-body{{line-height:1.8;color:#cbd5e1;font-size:.9rem}}
.coaching-body p{{margin-bottom:1rem}}
.coaching-body ul{{margin:.5rem 0 1rem 1.5rem}}
.coaching-body li{{margin-bottom:.4rem;list-style:none;display:flex;align-items:flex-start;gap:.5rem}}
.coaching-body strong{{color:#f1f5f9}}
.ch3{{font-size:.95rem;font-weight:700;color:#60a5fa;margin:1.25rem 0 .5rem;padding-bottom:.3rem;border-bottom:1px solid rgba(96,165,250,.15)}}
.step-num{{display:inline-flex;align-items:center;justify-content:center;min-width:1.4rem;height:1.4rem;background:#3b82f6;border-radius:50%;font-size:.7rem;font-weight:700;color:white;margin-right:.5rem}}
.footer{{padding:1.5rem 2.5rem;text-align:center;font-size:.72rem;color:#334155;font-family:'JetBrains Mono',monospace;background:#111827}}
@media print{{body{{background:white;padding:0}}.wrap{{box-shadow:none;border-radius:0}}.hdr{{print-color-adjust:exact;-webkit-print-color-adjust:exact}}img{{max-width:100%!important;print-color-adjust:exact;-webkit-print-color-adjust:exact}}}}
</style></head><body>
<div class="wrap">
  <div class="hdr">
    <div class="htop">
      <div class="logo"> FormAI</div>
      <div class="meta"><div>Generated: {timestamp}</div><div>Exercise: {exercise}</div><div>{'Session Report' if report_type=='session' else 'Frame Report'}</div></div>
    </div>
    <h1>Exercise Form Analysis Report</h1>
    <div class="badges">
      <span class="badge"> MediaPipe Pose Detection</span>
      <span class="badge"> Random Forest ML</span>
      <span class="badge"> Medical-Grade Feedback</span>
      <span class="badge"> Grad-CAM Heatmaps</span>
      <span class="badge"> Gemini AI Coaching</span>
    </div>
  </div>
  {current_html}
  {session_html}
  {coaching_html}
  <div class="footer">FormAI — AI-Powered Exercise Form Analysis &nbsp;|&nbsp; FYP Project &nbsp;|&nbsp; {timestamp}</div>
</div>
</body></html>"""

    return HTMLResponse(content=html)

@app.post("/detailed-report")
async def generate_detailed_report(data: dict):
    session_data = data.get("session", {})
    reps         = session_data.get("reps", [])
    summary      = session_data.get("summary", {})
    exercise     = session_data.get("exercise", "Bench Press")
    timestamp    = session_data.get("timestamp", "")

    GC = {"A":"#22c55e","B":"#3b82f6","C":"#eab308","D":"#f97316","F":"#ef4444","—":"#6b7280"}
    def gc(g): return GC.get(g,"#6b7280")

    def parse_report_to_html(txt):
        if not txt:
            return ""
        lines = txt.split("\n")
        html = []
        for line in lines:
            line = line.rstrip()
            if "=" * 10 in line:
                html.append('<hr style="border:1px solid rgba(255,255,255,0.08);margin:0.5rem 0"/>')
            elif line.startswith("ERROR #"):
                html.append(f'<div style="font-size:0.95rem;font-weight:700;color:#f97316;margin:0.75rem 0 0.25rem">{line}</div>')
            elif line.startswith("SEVERITY:"):
                sev = line.replace("SEVERITY:","").strip()
                col = "#ef4444" if sev=="CRITICAL" else "#f97316" if sev=="HIGH" else "#eab308" if sev=="MEDIUM" else "#94a3b8"
                html.append(f'<div style="font-size:0.8rem;margin-bottom:0.5rem;color:{col}">Severity: <strong>{sev}</strong></div>')
            elif any(line.startswith(h) for h in ["AFFECTED ANATOMY:","JOINTS AT RISK:","INJURY RISK FACTORS:","CORRECTION PROTOCOL:","RECOMMENDATIONS","IMPORTANT DISCLAIMER"]):
                html.append(f'<div style="font-size:0.75rem;font-weight:700;color:#60a5fa;text-transform:uppercase;letter-spacing:0.05em;margin:0.75rem 0 0.4rem;padding-top:0.5rem;border-top:1px solid rgba(255,255,255,0.06)">{line}</div>')
            elif line.startswith("  - ") and not line.startswith("    "):
                html.append(f'<div style="font-size:0.8rem;color:#cbd5e1;margin:0.2rem 0 0.1rem 0.5rem">• {line[4:]}</div>')
            elif line.startswith("    Function:") or line.startswith("    Location:") or line.startswith("    Risk Level:"):
                html.append(f'<div style="font-size:0.72rem;color:#64748b;margin-left:1rem">{line.strip()}</div>')
            elif line.strip() and line.strip()[0].isdigit() and ". " in line:
                html.append(f'<div style="font-size:0.8rem;color:#94a3b8;margin:0.2rem 0 0.1rem 0.5rem">{line.strip()}</div>')
            elif line.strip() and "=" * 5 not in line:
                html.append(f'<div style="font-size:0.8rem;color:#94a3b8;margin:0.1rem 0">{line}</div>')
        return "".join(html)

    # Build rep sections
    rep_sections = ""
    for rep in reps:
        grade   = rep.get("grade","—")
        errors  = rep.get("errors",[])
        heatmap = rep.get("heatmap")
        detailed_txt = rep.get("detailed_report","")

        error_html = ""
        for e in errors:
            sev_color = "#ef4444" if e.get("severity")=="CRITICAL" else "#f97316" if e.get("severity")=="HIGH" else "#eab308"
            muscles = ", ".join(e.get("muscles",[]))
            error_html += f'''
            <div style="background:rgba(239,68,68,0.08);border-left:4px solid {sev_color};border-radius:8px;padding:0.875rem;margin-bottom:0.5rem">
              <div style="font-weight:600;font-size:0.9rem;color:{sev_color};margin-bottom:0.25rem">{e.get("type","")}</div>
              <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:0.2rem">Severity: <strong style="color:{sev_color}">{e.get("severity","")}</strong></div>
              <div style="font-size:0.78rem;color:#64748b">Affected: {muscles}</div>
            </div>'''

        detailed_html = ""
        if detailed_txt:
            detailed_html = f'<div style="margin-top:0.75rem;padding:1rem;background:#0a0d14;border-radius:8px;border:1px solid rgba(255,255,255,0.06)">{parse_report_to_html(detailed_txt)}</div>'

        heatmap_html = ""
        if heatmap:
            heatmap_html = f'''
            <div style="margin-top:1rem;background:#0f1117;border-radius:8px;padding:0.75rem;border:1px solid rgba(239,68,68,0.25)">
              <div style="font-size:0.7rem;color:#f97316;font-weight:600;margin-bottom:0.5rem;text-transform:uppercase;letter-spacing:0.05em">
                 Grad-CAM Heatmap — Rep #{rep.get("rep_number","?")}
              </div>
              <img src="{heatmap}" style="width:100%;border-radius:6px"/>
              <div style="font-size:0.65rem;color:#475569;margin-top:0.4rem">Blue = Safe | Yellow = Caution | Red = High Injury Risk</div>
            </div>'''

        tech_color = "#22c55e" if grade != "F" else "#ef4444"
        tech_text  = " Correct" if grade != "F" else " Incorrect"

        rep_sections += f'''
        <div style="background:#1a1f2e;border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:1.25rem;margin-bottom:1.5rem">
          <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin-bottom:1rem;padding-bottom:1rem;border-bottom:1px solid rgba(255,255,255,0.06)">
            <div style="font-size:1rem;font-weight:700;color:#f1f5f9">Rep #{rep.get("rep_number","?")}</div>
            <div style="display:inline-flex;align-items:center;justify-content:center;width:2rem;height:2rem;border-radius:6px;font-weight:700;color:white;background:{gc(grade)}">{grade}</div>
            <div style="font-size:0.85rem;font-weight:600;color:{tech_color}">{tech_text}</div>
            <div style="font-size:0.75rem;color:#64748b;font-family:monospace;margin-left:auto">Elbow: {rep.get("elbow_angle","—")}° | Symmetry: {rep.get("symmetry_grade","—")} | Phase: {rep.get("phase","—")}</div>
          </div>
          {f'<div><div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.75rem;font-weight:600"> Issues Detected</div>{error_html}</div>' if error_html else '<div style="padding:0.75rem;background:rgba(34,197,94,0.08);border-radius:8px;color:#86efac;font-size:0.85rem"> No issues detected</div>'}
          {detailed_html}
          {heatmap_html}
        </div>'''

    total   = summary.get("total_reps",0)
    correct = summary.get("correct_reps",0)
    acc     = summary.get("accuracy",0)
    avg_el  = summary.get("avg_elbow","—")
    grades  = summary.get("grade_distribution",{})
    heatmap_count = sum(1 for r in reps if r.get("heatmap"))
    gdist = "".join(f'<span style="display:inline-flex;align-items:center;justify-content:center;padding:0.2rem 0.75rem;border-radius:6px;font-weight:700;font-size:0.8rem;color:white;margin-right:0.5rem;background:{gc(g)}">{g}: {c}</span>' for g,c in grades.items() if c>0)

    all_errors = [e for r in reps for e in r.get("errors",[])]
    has_critical = any(e.get("severity")=="CRITICAL" for e in all_errors)
    has_high     = any(e.get("severity")=="HIGH" for e in all_errors)

    if has_critical:
        rec_color = "#ef4444"; rec_title = " CRITICAL ISSUES DETECTED"
        rec_items = ["Stop current set immediately","Reduce weight by 40-50%","Focus on technique correction before adding weight","Consider consulting with a certified professional"]
    elif has_high:
        rec_color = "#f97316"; rec_title = " HIGH-RISK ISSUES DETECTED"
        rec_items = ["Reduce weight by 20-30%","Address form issues before progressing","Film your sets to monitor improvement"]
    else:
        rec_color = "#eab308"; rec_title = "ℹ MODERATE ISSUES DETECTED"
        rec_items = ["Make corrections with current weight","Focus on quality movement patterns","Monitor for improvement over next few sessions"]

    rec_html = "".join(f"<li style='color:#cbd5e1;font-size:0.875rem;margin-bottom:0.4rem'>{item}</li>" for item in rec_items)

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>FormAI Biomechanical Report</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Space Grotesk',sans-serif;background:#0f1117;color:#e2e8f0;padding:2rem}}
.wrap{{max-width:900px;margin:0 auto}}
@media print{{body{{background:white;padding:0}}img{{max-width:100%!important;print-color-adjust:exact;-webkit-print-color-adjust:exact}}}}
</style></head><body>
<div class="wrap">
  <div style="background:linear-gradient(135deg,#1e3a5f,#0f2744);padding:2rem;border-radius:16px;margin-bottom:2rem">
    <div style="font-size:1.8rem;font-weight:700;background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.5rem"> FormAI</div>
    <h1 style="font-size:1.4rem;color:#f1f5f9;margin-bottom:0.5rem">Biomechanical Analysis Report</h1>
    <div style="font-size:0.75rem;color:#64748b;font-family:'JetBrains Mono',monospace">Exercise: {exercise} | Generated: {timestamp}</div>
  </div>

  <div style="background:#1a1f2e;border-radius:12px;padding:1.5rem;margin-bottom:2rem;border:1px solid rgba(255,255,255,0.06)">
    <div style="font-size:0.8rem;color:#60a5fa;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:1rem;font-weight:700">Session Summary</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:0.75rem;margin-bottom:1rem">
      <div style="background:#232936;border-radius:8px;padding:0.75rem;text-align:center"><div style="font-size:1.8rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:#f1f5f9">{total}</div><div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;margin-top:0.25rem">Total Reps</div></div>
      <div style="background:#232936;border-radius:8px;padding:0.75rem;text-align:center"><div style="font-size:1.8rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:#22c55e">{correct}</div><div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;margin-top:0.25rem">Correct Reps</div></div>
      <div style="background:#232936;border-radius:8px;padding:0.75rem;text-align:center"><div style="font-size:1.8rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:'#22c55e' if acc>=80 else '#f97316'">{acc:.1f}%</div><div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;margin-top:0.25rem">Accuracy</div></div>
      <div style="background:#232936;border-radius:8px;padding:0.75rem;text-align:center"><div style="font-size:1.8rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:#f1f5f9">{avg_el}°</div><div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;margin-top:0.25rem">Avg Elbow</div></div>
      <div style="background:#232936;border-radius:8px;padding:0.75rem;text-align:center"><div style="font-size:1.8rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:#f97316">{heatmap_count}</div><div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;margin-top:0.25rem">Heatmaps</div></div>
    </div>
    <div>{gdist}</div>
  </div>

  <div style="background:#1a1f2e;border-radius:12px;padding:1.5rem;margin-bottom:2rem;border-left:4px solid {rec_color}">
    <h2 style="color:{rec_color};font-size:1rem;margin-bottom:1rem">{rec_title}</h2>
    <ul style="list-style:none">{rec_html}</ul>
  </div>

  <div style="font-size:0.8rem;color:#60a5fa;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:1rem;font-weight:700">Rep-by-Rep Analysis</div>
  {rep_sections}

  <div style="background:#111827;border-radius:12px;padding:1.25rem;font-size:0.78rem;color:#475569;line-height:1.6">
    <strong style="color:#64748b"> Medical Disclaimer</strong><br/><br/>
    This analysis is based on computer vision and biomechanical algorithms. For personalized medical advice, diagnosis, or treatment of injuries, always consult with a Licensed Physical Therapist, Sports Medicine Physician, or Certified Strength and Conditioning Specialist (CSCS).<br/><br/>
    If you experience pain, stop immediately and seek professional medical evaluation.
  </div>

  <div style="text-align:center;font-size:0.7rem;color:#334155;margin-top:2rem;font-family:'JetBrains Mono',monospace">FormAI — Biomechanical Analysis Report | {timestamp}</div>
</div>
</body></html>"""

    return HTMLResponse(content=html)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
