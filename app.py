import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from datetime import datetime
import json
import time
from collections import deque
import threading
import sys
import signal

app = Flask(__name__)

# Configuration
MIN_CONTOUR_AREA = 500
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CANNY_THRESHOLDS = (50, 150)

# Alert thresholds
RISK_LEVELS = {
    'SAFE': {'color': (0, 255, 0), 'threshold': 0},
    'LOW': {'color': (0, 255, 255), 'threshold': 3},
    'MEDIUM': {'color': (0, 165, 255), 'threshold': 8},
    'HIGH': {'color': (0, 0, 255), 'threshold': 15},
    'CRITICAL': {'color': (255, 0, 255), 'threshold': 25}
}

# Global state management
class MonitoringState:
    def __init__(self):
        self.lock = threading.Lock()
        self.stability_history = deque(maxlen=100)  # Last 100 frames
        self.alert_history = []
        self.zone_data = {}
        self.start_time = datetime.now()
        self.total_frames = 0
        self.unstable_frames = 0
        self.current_risk_level = 'SAFE'
        self.emergency_alerts = []

state = MonitoringState()

def calculate_risk_score(contours):
    """Calculate risk score based on contour analysis (optimized)"""
    if not contours:
        return 0

    areas = [cv2.contourArea(cnt) for cnt in contours]
    large_areas = [a for a in areas if a > MIN_CONTOUR_AREA]
    total_area = sum(large_areas)
    num_large_contours = len(large_areas)

    # Risk factors
    area_factor = min(total_area / 50000.0, 1.0) * 15
    count_factor = min(num_large_contours / 10.0, 1.0) * 10

    return area_factor + count_factor

def determine_risk_level(risk_score):
    """Determine risk level based on score"""
    if risk_score >= RISK_LEVELS['CRITICAL']['threshold']:
        return 'CRITICAL'
    elif risk_score >= RISK_LEVELS['HIGH']['threshold']:
        return 'HIGH'
    elif risk_score >= RISK_LEVELS['MEDIUM']['threshold']:
        return 'MEDIUM'
    elif risk_score >= RISK_LEVELS['LOW']['threshold']:
        return 'LOW'
    return 'SAFE'

def create_alert(risk_level, risk_score, zones):
    """Create emergency alert"""
    alert = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'risk_level': risk_level,
        'risk_score': round(risk_score, 2),
        'affected_zones': zones,
        'message': generate_alert_message(risk_level)
    }

    with state.lock:
        state.alert_history.append(alert)
        if len(state.alert_history) > 50:
            state.alert_history.pop(0)

        if risk_level in ['HIGH', 'CRITICAL']:
            state.emergency_alerts.append(alert)
            if len(state.emergency_alerts) > 20:
                state.emergency_alerts.pop(0)

    return alert

def generate_alert_message(risk_level):
    """Generate appropriate alert message"""
    messages = {
        'SAFE': 'Terrain is stable and safe for operations',
        'LOW': 'Minor terrain irregularities detected',
        'MEDIUM': 'Moderate instability - Exercise caution',
        'HIGH': 'High risk terrain - Immediate attention required',
        'CRITICAL': 'CRITICAL - Evacuate area immediately!'
    }
    return messages.get(risk_level, 'Unknown status')

def analyze_zones(frame, contours):
    """Divide frame into zones and analyze each"""
    h, w = frame.shape[:2]
    zone_width = w // 3
    zone_height = h // 3

    zones = {}
    for row in range(3):
        for col in range(3):
            zone_id = f"Z{row}{col}"
            x1, y1 = col * zone_width, row * zone_height
            x2, y2 = x1 + zone_width, y1 + zone_height

            zone_unstable = False
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > MIN_CONTOUR_AREA:
                    M = cv2.moments(cnt)
                    if M.get("m00", 0) != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        if x1 <= cx < x2 and y1 <= cy < y2:
                            zone_unstable = True
                            break

            zones[zone_id] = {
                'status': 'UNSAFE' if zone_unstable else 'SAFE',
                'coords': (x1, y1, x2, y2)
            }

    return zones

def draw_enhanced_overlay(frame, contours, risk_level, risk_score, zones):
    """Draw comprehensive visualization overlay"""
    h, w = frame.shape[:2]

    # Draw grid zones
    zone_width = w // 3
    zone_height = h // 3

    for row in range(3):
        for col in range(3):
            zone_id = f"Z{row}{col}"
            x1, y1 = col * zone_width, row * zone_height
            x2, y2 = x1 + zone_width, y1 + zone_height

            if zones.get(zone_id, {}).get('status') == 'UNSAFE':
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Draw contours
    for contour in contours:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)

    # Status panel
    panel_height = 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Risk level indicator
    risk_color = RISK_LEVELS[risk_level]['color']
    cv2.rectangle(frame, (10, 10), (200, 45), risk_color, -1)
    cv2.putText(frame, f"RISK: {risk_level}", (20, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)

    # Risk score
    cv2.putText(frame, f"Score: {risk_score:.1f}", (220, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Statistics (use lock when reading shared state)
    with state.lock:
        total = max(state.total_frames, 1)
        stability_rate = ((state.total_frames - state.unstable_frames) / total) * 100
    cv2.putText(frame, f"Stability: {stability_rate:.1f}%", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Unsafe zones count
    unsafe_count = sum(1 for z in zones.values() if z['status'] == 'UNSAFE')
    cv2.putText(frame, f"Unsafe Zones: {unsafe_count}/9", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Timestamp
    timestamp = datetime.now().strftime('%H:%M:%S')
    cv2.putText(frame, timestamp, (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Critical alert flash
    if risk_level == 'CRITICAL':
        if int(time.time() * 2) % 2:  # Blink effect
            cv2.rectangle(frame, (5, 5), (w-5, h-5), (255, 0, 255), 5)

    return frame

def analyze_terrain(frame):
    """Enhanced terrain analysis with risk assessment"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_THRESHOLDS[0], CANNY_THRESHOLDS[1])
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    risk_score = calculate_risk_score(contours)
    risk_level = determine_risk_level(risk_score)
    zones = analyze_zones(frame, contours)

    is_stable = (risk_level == 'SAFE')

    # Update statistics (use lock)
    with state.lock:
        state.total_frames += 1
        if not is_stable:
            state.unstable_frames += 1

    # Record history
    with state.lock:
        state.stability_history.append({
            'timestamp': time.time(),
            'stable': is_stable,
            'risk_score': risk_score,
            'risk_level': risk_level
        })



        




        

    # Create alert if risk changed or is high
    create_needed = False
    with state.lock:
        if risk_level != state.current_risk_level or risk_level in ['HIGH', 'CRITICAL']:
            create_needed = True
            state.current_risk_level = risk_level

    if create_needed:
        unsafe_zones = [zid for zid, zdata in zones.items() if zdata['status'] == 'UNSAFE']
        create_alert(risk_level, risk_score, unsafe_zones)

    # update zone data (protected)
    with state.lock:
        state.zone_data = zones

    return is_stable, contours, risk_level, risk_score, zones

def generate_frames():
    cap = cv2.VideoCapture("videos/2nd.mp4")
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        while True:
            error_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
            cv2.putText(error_frame, "CAMERA ERROR", (50, FRAME_HEIGHT//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', error_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                time.sleep(0.1)
                continue
  


            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Enhanced analysis
            is_stable, contours, risk_level, risk_score, zones = analyze_terrain(frame)

            # Enhanced visualization
            frame = draw_enhanced_overlay(frame, contours, risk_level, risk_score, zones)

            # Terminal output
            if is_stable:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Terrain STABLE - Risk: {risk_score:.1f}")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: {risk_level} Risk - Score: {risk_score:.1f}")

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except GeneratorExit:
        # Client disconnected - clean up
        pass
    except Exception as e:
        print("Camera loop exception:", e)
    finally:
        cap.release()
        print("Camera released.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    """Real-time status API"""
    with state.lock:
        recent_history = list(state.stability_history)[-30:]
        zones = dict(state.zone_data)
        total = state.total_frames
        unstable = state.unstable_frames
        current = state.current_risk_level
        uptime = str(datetime.now() - state.start_time).split('.')[0]

    return jsonify({
        'current_risk': current,
        'total_frames': total,
        'unstable_frames': unstable,
        'stability_rate': round(((total - unstable) / max(total, 1)) * 100, 2),
        'zones': zones,
        'recent_history': recent_history,
        'uptime': uptime
    })

@app.route('/api/alerts')
def api_alerts():
    """Get alert history"""
    with state.lock:
        all_alerts = list(state.alert_history[-20:])
        emergency = list(state.emergency_alerts[-10:])
    return jsonify({
        'all_alerts': all_alerts,
        'emergency_alerts': emergency
    })

@app.route('/api/historical')
def api_historical():
    """Get historical stability data"""
    with state.lock:
        history = list(state.stability_history)

    # Aggregate by minute for chart
    aggregated = {}
    for record in history:
        minute = int(record['timestamp'] / 60) * 60
        if minute not in aggregated:
            aggregated[minute] = {'stable': 0, 'unstable': 0, 'total': 0, 'avg_risk': 0}

        aggregated[minute]['total'] += 1
        aggregated[minute]['avg_risk'] += record['risk_score']
        if record['stable']:
            aggregated[minute]['stable'] += 1
        else:
            aggregated[minute]['unstable'] += 1

    chart_data = []
    for timestamp, data in sorted(aggregated.items()):
        chart_data.append({
            'time': datetime.fromtimestamp(timestamp).strftime('%H:%M'),
            'stability_rate': round((data['stable'] / data['total']) * 100, 1),
            'avg_risk_score': round(data['avg_risk'] / data['total'], 2)
        })

    return jsonify(chart_data)

@app.route('/api/reset')
def api_reset():
    """Reset monitoring statistics"""
    global state
    with state.lock:
        state = MonitoringState()
    return jsonify({'status': 'reset_complete'})

def handle_sigint(signum, frame):
    print("Received termination signal; exiting.")
    sys.exit(0)

if __name__ == '__main__':
    # Prevent Flask dev reloader from opening camera twice.
    print("=" * 60)
    print("DISASTER MANAGEMENT TERRAIN MONITORING SYSTEM")
    print("Real-Time Safety Assessment Platform")
    print("=" * 60)
    print(f"Starting server at http://localhost:5000")
    print(f"System initialized: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    signal.signal(signal.SIGINT, handle_sigint)
    # IMPORTANT: set debug=False to avoid double-run / camera lock issues while developing
    app.run(host='0.0.0.0', port=5000, debug=False)
