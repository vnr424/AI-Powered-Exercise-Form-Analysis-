'use client'

import { useState, useRef } from 'react'
import { Video, VideoOff, Activity, Download, FileText } from 'lucide-react'

const EXERCISES = [
  { key: 'bench_press',    label: 'Bench Press',             icon: '' },
  { key: 'shoulder_press', label: 'Standing Shoulder Press', icon: '' },
]

export default function WebcamAnalysis() {
  const [isActive, setIsActive]                 = useState(false)
  const [countdown, setCountdown]               = useState<number|null>(null)
  const [results, setResults]                   = useState<any>(null)
  const [status, setStatus]                     = useState('Stopped')
  const [downloading, setDownloading]           = useState(false)
  const [selectedExercise, setSelectedExercise] = useState(EXERCISES[0])
  const [activeExercise, setActiveExercise]     = useState(EXERCISES[0])
  const videoRef    = useRef<HTMLVideoElement>(null)
  const canvasRef   = useRef<HTMLCanvasElement>(null)
  const overlayRef  = useRef<HTMLCanvasElement>(null)
  const wsRef       = useRef<WebSocket | null>(null)
  const streamRef   = useRef<MediaStream | null>(null)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const sessionRepsRef    = useRef<any[]>([])
  const currentResultsRef = useRef<any>(null)
  const WEBSOCKET_URL = 'ws://localhost:8000/ws'
  const API_URL       = 'http://localhost:8000'
  const CONNECTIONS   = [[11,13],[13,15],[12,14],[14,16],[11,12],[11,23],[12,24],[23,24]]

  const drawLandmarks = (landmarks: any[], videoEl: HTMLVideoElement) => {
    const canvas = overlayRef.current
    if (!canvas || !landmarks || landmarks.length === 0) return
    canvas.width  = videoEl.videoWidth  || videoEl.clientWidth
    canvas.height = videoEl.videoHeight || videoEl.clientHeight
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    const lmMap: Record<number,any> = {}
    landmarks.forEach((lm:any) => { lmMap[lm.index] = lm })
    CONNECTIONS.forEach(([a,b]) => {
      const p1=lmMap[a], p2=lmMap[b]
      if (!p1||!p2||p1.visibility<0.3||p2.visibility<0.3) return
      ctx.beginPath(); ctx.moveTo(p1.x,p1.y); ctx.lineTo(p2.x,p2.y)
      ctx.strokeStyle='rgba(0,255,255,0.85)'; ctx.lineWidth=3; ctx.stroke()
    })
    landmarks.forEach((lm:any) => {
      if (lm.visibility<0.3) return
      const isElbow = lm.index===13||lm.index===14
      ctx.beginPath(); ctx.arc(lm.x,lm.y,isElbow?8:6,0,2*Math.PI)
      ctx.fillStyle=isElbow?'rgba(255,165,0,0.95)':'rgba(0,255,0,0.9)'
      ctx.strokeStyle='white'; ctx.lineWidth=2; ctx.fill(); ctx.stroke()
    })
  }

  const switchExercise = (exercise: typeof EXERCISES[0]) => {
    setSelectedExercise(exercise)
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'set_exercise', exercise: exercise.key }))
      setActiveExercise(exercise)
      sessionRepsRef.current = []
      setResults(null)
    }
  }

  const startCountdown = () => {
    setCountdown(5)
    let count = 5
    const timer = setInterval(() => {
      count -= 1
      if (count <= 0) {
        clearInterval(timer)
        setCountdown(null)
        startCamera()
      } else {
        setCountdown(count)
      }
    }, 1000)
  }

  const startCamera = async () => {
    try {
      setStatus('Starting camera...')
      sessionRepsRef.current = []
      const stream = await navigator.mediaDevices.getUserMedia({ video:{width:640,height:480} })
      if (videoRef.current) { videoRef.current.srcObject=stream; streamRef.current=stream }
      setStatus('Connecting...')
      const ws = new WebSocket(WEBSOCKET_URL)
      ws.onopen = () => {
        setStatus('Connected'); setIsActive(true)
        ws.send(JSON.stringify({ type: 'set_exercise', exercise: selectedExercise.key }))
        setActiveExercise(selectedExercise)
        startSendingFrames(ws)
      }
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        if (data.type === 'exercise_set') return
        setResults(data); currentResultsRef.current = data
        if (data.rep_just_completed && data.rep_count > 0) {
          const repGrade = data.grade; const techCorrect = !['F'].includes(repGrade); sessionRepsRef.current.push({ rep_number:data.rep_count, grade:repGrade, prediction:techCorrect?1:0, elbow_angle:data.elbow_angle, symmetry_grade:data.symmetry_grade, phase:data.phase, errors:data.errors||[], heatmap:data.heatmap||null, detailed_report:data.detailed_report||'' })
        }
        if (data.landmarks && videoRef.current) drawLandmarks(data.landmarks, videoRef.current)
      }
      ws.onerror = () => setStatus('Connection Error')
      ws.onclose = () => { setStatus('Disconnected'); stopCamera() }
      wsRef.current = ws
    } catch { alert('Camera access denied!'); setStatus('Error') }
  }

  const startSendingFrames = (ws: WebSocket) => {
    const interval = setInterval(() => {
      if (canvasRef.current && videoRef.current && ws.readyState===WebSocket.OPEN) {
        const canvas=canvasRef.current, video=videoRef.current
        if (video.readyState!==video.HAVE_ENOUGH_DATA) return
        const ctx=canvas.getContext('2d')
        if (ctx) { canvas.width=video.videoWidth; canvas.height=video.videoHeight; ctx.drawImage(video,0,0); ws.send(JSON.stringify({ frame: canvas.toDataURL('image/jpeg',0.5) })) }
      }
    }, 200)
    intervalRef.current = interval
  }

  const stopCamera = () => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    if (streamRef.current)   streamRef.current.getTracks().forEach(t=>t.stop())
    if (wsRef.current)       wsRef.current.close()
    if (overlayRef.current) { const ctx=overlayRef.current.getContext('2d'); if (ctx) ctx.clearRect(0,0,overlayRef.current.width,overlayRef.current.height) }
    setIsActive(false); setStatus('Stopped')
  }

  const buildSessionSummary = () => {
    const reps = sessionRepsRef.current
    if (reps.length === 0) return {}
    const correct=reps.filter((r:any)=>r.prediction===1).length
    const grades: Record<string,number>={A:0,B:0,C:0,D:0,F:0}
    reps.forEach((r:any)=>{if(grades[r.grade]!==undefined)grades[r.grade]++})
    return { total_reps:reps.length, correct_reps:correct, accuracy:(correct/reps.length)*100, avg_elbow:Math.round(reps.reduce((s:number,r:any)=>s+(r.elbow_angle||0),0)/reps.length*10)/10, grade_distribution:grades }
  }

  const openReport = async (type: 'frame'|'session') => {
    if (type==='frame'&&!currentResultsRef.current){alert('No analysis data yet!');return}
    if (type==='session'&&sessionRepsRef.current.length===0){alert('No reps completed yet!');return}
    setDownloading(true)
    try {
      let coachingHTML=''
      if (type==='session_ai'&&sessionRepsRef.current.length>0) {
        try {
          const errorCounts: Record<string,number>={}
          sessionRepsRef.current.forEach((rep:any)=>{ rep.errors?.forEach((e:any)=>{ errorCounts[e.type]=(errorCounts[e.type]||0)+1 }) })
          const r=await fetch('/api/coaching',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({errors:errorCounts,stats:buildSessionSummary(),exercise:activeExercise.label})})
          coachingHTML=(await r.json()).coaching||''
        } catch(e){console.error(e)}
      }
      const body=type==='frame'
        ?{type:'frame',session:{current:currentResultsRef.current,exercise:activeExercise.label,timestamp:new Date().toLocaleString()}}
        :{type:'session',session:{reps:sessionRepsRef.current,summary:buildSessionSummary(),current:currentResultsRef.current,exercise:activeExercise.label,timestamp:new Date().toLocaleString(),coaching:coachingHTML}}
      const endpoint = type==='session_ai' ? `${API_URL}/report` : type==='session' ? `${API_URL}/detailed-report` : `${API_URL}/report`
      const res=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
      const html=await res.text()
      const win=window.open('','_blank')
      if(!win){alert('Allow popups!');setDownloading(false);return}
      win.document.write(html); win.document.close()
      setTimeout(()=>{win.focus();win.print()},800)
    } catch{alert('Could not generate report. Is the server running?')}
    setDownloading(false)
  }

  const getGradeColor=(g:string)=>({'A':'text-green-400','B':'text-blue-400','C':'text-yellow-400','D':'text-orange-400','F':'text-red-400','—':'text-gray-400'}[g]||'text-gray-400')
  const getPhaseColor=(p:string)=>({'TOP':'bg-green-600','BOTTOM':'bg-blue-600','MOVING_UP':'bg-yellow-600','MOVING_DOWN':'bg-cyan-600','UNKNOWN':'bg-gray-600'}[p]||'bg-gray-600')

  return (
    <section id="live-demo" className="py-20 bg-gradient-to-br from-gray-900 to-gray-800 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8">
          <h2 className="text-4xl font-bold mb-4">Try It Live!</h2>
          <p className="text-xl text-gray-300">Select your exercise and get real-time AI biomechanical feedback</p>
          {status==='Connected'
            ?<p className="text-sm text-green-400 mt-2">Real-time analysis active — {activeExercise.label}</p>
            :<p className="text-sm text-yellow-400 mt-2">Analysis server must be running on host machine</p>}
        </div>
        <div className="flex justify-center mb-8">
          <div className="bg-gray-900 rounded-2xl p-2 flex gap-2 border border-gray-700">
            {EXERCISES.map((ex)=>(
              <button key={ex.key} onClick={()=>switchExercise(ex)}
                className={`flex items-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all duration-200 ${selectedExercise.key===ex.key?'bg-blue-600 text-white shadow-lg scale-105':'text-gray-400 hover:text-white hover:bg-gray-800'}`}>
                <span className="text-xl">{ex.icon}</span>
                <span>{ex.label}</span>
                {isActive&&activeExercise.key===ex.key&&<span className="w-2 h-2 bg-green-400 rounded-full animate-pulse ml-1"/>}
              </button>
            ))}
          </div>
        </div>
        <div className="max-w-2xl mx-auto mb-8">
          {selectedExercise.key==='bench_press'
            ?<div className="bg-blue-900/30 border border-blue-700 rounded-xl p-4 text-sm text-blue-200 text-center"><strong>Bench Press:</strong> Keep elbows at 45–75° • Retract shoulder blades • Even bar path</div>
            :<div className="bg-purple-900/30 border border-purple-700 rounded-xl p-4 text-sm text-purple-200 text-center"><strong>Shoulder Press:</strong> Forearms vertical at start • Elbows at 60–90° • Engage core • No hip lean</div>}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="bg-gray-900 rounded-2xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold">Your Camera</h3>
              
            </div>
            <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
              <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover"/>
              <canvas ref={overlayRef} className="absolute inset-0 w-full h-full" style={{pointerEvents:'none'}}/>
              <canvas ref={canvasRef} className="hidden"/>
              {countdown!==null&&<div className="absolute inset-0 flex items-center justify-center bg-black/80 z-20"><div className="text-center"><div className="text-8xl font-bold text-white mb-4 animate-pulse">{countdown}</div><p className="text-gray-300 text-lg">Get in position!</p><p className="text-gray-400 text-sm mt-2">Analysis starts soon...</p></div></div>}
              {!isActive&&countdown===null&&<div className="absolute inset-0 flex items-center justify-center bg-black"><div className="text-center"><VideoOff className="w-16 h-16 text-gray-600 mx-auto mb-3"/><p className="text-gray-500 text-sm"> Ready for {selectedExercise.label}</p></div></div>}
              {results?.phase&&results.phase!=='UNKNOWN'&&<div className={`absolute top-3 left-3 px-3 py-1 rounded-full text-xs font-bold ${getPhaseColor(results.phase)}`}>{results.phase}</div>}
              {results?.confidence&&<div className="absolute top-3 right-3 px-3 py-1 bg-black/60 rounded-full text-xs font-bold">{results.confidence}% conf</div>}
            </div>
            <div className="mt-4">
              {!isActive
                ?<button onClick={startCountdown} disabled={countdown!==null} className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 text-white px-6 py-3 rounded-lg font-semibold transition flex items-center justify-center gap-2"><Video className="w-5 h-5"/>{countdown!==null ? `Starting in ${countdown}...` : `Start ${selectedExercise.label} Analysis`}</button>
                :<button onClick={stopCamera}  className="w-full bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-lg font-semibold transition flex items-center justify-center gap-2"><VideoOff className="w-5 h-5"/>Stop Analysis</button>}
            </div>
            <div className="mt-3 text-center">
              <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm ${status==='Connected'?'bg-green-600':status.includes('...')?'bg-yellow-600':status==='Connection Error'?'bg-red-600':'bg-gray-600'}`}>
                <Activity className={`w-4 h-4 mr-2 ${status==='Connected'?'animate-pulse':''}`}/>{status}
              </span>
            </div>
            {isActive&&(
              <div className="mt-4 border-t border-gray-700 pt-4">
                <p className="text-xs text-gray-500 mb-2 text-center">Switch exercise (resets rep count)</p>
                <div className="flex gap-2">
                  {EXERCISES.map((ex)=>(
                    <button key={ex.key} onClick={()=>switchExercise(ex)}
                      className={`flex-1 py-2 rounded-lg text-xs font-semibold transition ${activeExercise.key===ex.key?'bg-blue-600 text-white':'bg-gray-800 text-gray-400 hover:bg-gray-600'}`}>
                      {ex.icon} {ex.label}
                    </button>
                  ))}
                </div>
              </div>
            )}
            <div className="mt-4 border-t border-gray-700 pt-4">
              <p className="text-xs text-gray-500 mb-3 text-center font-semibold uppercase tracking-wider"> Download PDF Reports</p>
              <div className="flex flex-col gap-3">
                <button onClick={()=>openReport('session')} disabled={sessionRepsRef.current.length===0||downloading} className="flex items-center justify-center gap-2 px-4 py-3 rounded-xl text-sm font-semibold bg-purple-600 hover:bg-purple-500 disabled:bg-gray-800 disabled:text-gray-500 disabled:cursor-not-allowed transition"><Download className="w-4 h-4"/>{downloading?'Generating...':`Session Report${sessionRepsRef.current.length>0?` (${sessionRepsRef.current.length} reps)`:''}`}</button>
                <button onClick={()=>openReport('session_ai')} disabled={sessionRepsRef.current.length===0||downloading} className="flex items-center justify-center gap-2 px-4 py-3 rounded-xl text-sm font-semibold bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 disabled:bg-gray-800 disabled:text-gray-500 disabled:cursor-not-allowed transition"><Download className="w-4 h-4"/>{downloading?'Generating...':'Session + Gemini AI'}</button>
              </div>
              <p className="text-xs text-gray-600 mt-2 text-center">AI coaching analyses your full session</p>
            </div>
          </div>
          <div className="bg-gray-900 rounded-2xl p-6">
            <h3 className="text-xl font-bold mb-4">Real-Time Feedback</h3>
            {results&&results.status!=='no_person'?(
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-gray-800 rounded-lg p-4"><div className="text-sm text-gray-400 mb-1">Form Grade</div><div className={`text-4xl font-bold ${getGradeColor(results.grade)}`}>{results.grade}</div></div>
                  <div className="bg-gray-800 rounded-lg p-4"><div className="text-sm text-gray-400 mb-1">Rep Count</div><div className="text-4xl font-bold text-green-400">{results.rep_count}</div></div>
                </div>
                <div className="bg-gray-800 rounded-lg p-3 flex items-center justify-between">
                  <span className="text-sm text-gray-400">Technique</span>
                  {results.prediction===1?<span className="px-2 py-1 bg-green-600 rounded text-xs font-bold"> CORRECT</span>:<span className="px-2 py-1 bg-red-600 rounded text-xs font-bold"> INCORRECT</span>}
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <div className="text-sm text-gray-400 mb-2">Elbow Angles</div>
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div><div className="text-xs text-gray-500">Left</div><div className="text-lg font-bold text-orange-400">{results.left_elbow}°</div></div>
                    <div><div className="text-xs text-gray-500">Avg</div><div className="text-xl font-bold">{results.elbow_angle}°</div></div>
                    <div><div className="text-xs text-gray-500">Right</div><div className="text-lg font-bold text-orange-400">{results.right_elbow}°</div></div>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-gray-800 rounded-lg p-3"><div className="text-xs text-gray-400 mb-1">Symmetry</div><div className={`text-2xl font-bold ${getGradeColor(results.symmetry_grade)}`}>{results.symmetry_grade}</div><div className="text-xs text-gray-500">{results.symmetry_score}</div></div>
                  <div className="bg-gray-800 rounded-lg p-3"><div className="text-xs text-gray-400 mb-1">Bar Level</div><div className={`text-sm font-bold ${results.bar_level==='LEVEL'?'text-green-400':'text-red-400'}`}>{results.bar_level==='LEVEL'?' LEVEL':' UNEVEN'}</div></div>
                </div>
                <div className="bg-gray-800 rounded-lg p-4"><div className="text-sm text-gray-400 mb-2">AI Feedback</div><p className="text-gray-200 text-sm">{results.feedback}</p></div>
                {results.errors?.length>0&&(
                  <div className="bg-red-900/40 border border-red-600 rounded-lg p-3">
                    <div className="text-sm text-red-400 font-bold mb-2"> Issues Detected</div>
                    {results.errors.map((err:any,i:number)=>(
                      <div key={i} className="text-xs text-gray-300 mb-1"><span className="text-orange-400 font-semibold">{err.type}</span><span className="text-gray-500"> — {err.severity}</span></div>
                    ))}
                  </div>
                )}
                {results.heatmap&&(
                  <div className="bg-gray-800 rounded-lg p-3">
                    <div className="text-xs text-gray-400 mb-2"> Grad-CAM Heatmap</div>
                    <img src={results.heatmap} alt="Heatmap" className="w-full rounded-lg"/>
                    <p className="text-xs text-gray-500 mt-1 text-center">Blue = safe · Red = injury risk</p>
                  </div>
                )}
              </div>
            ):results?.status==='no_person'?(
              <div className="h-full flex items-center justify-center"><div className="text-center"><Activity className="w-16 h-16 mx-auto mb-4 opacity-50 text-gray-500"/><p className="text-yellow-400">No person detected</p><p className="text-sm mt-2 text-gray-500">Adjust your camera</p></div></div>
            ):(
              <div className="h-full flex items-center justify-center"><div className="text-center"><Activity className="w-16 h-16 mx-auto mb-4 opacity-50 text-gray-500"/><p className="text-gray-500">Start analysis to see live results</p><p className="text-sm mt-2 text-gray-600">{selectedExercise.icon} {selectedExercise.label}</p></div></div>
            )}
          </div>
        </div>
        {status==='Connected'&&(
          <div className="mt-8 bg-green-900/30 border border-green-600 rounded-lg p-4">
            <p className="text-green-200 text-sm"><strong> Live:</strong> MediaPipe BlazePose • Random Forest ML (93.28% accuracy) • Biomechanical analysis • Grad-CAM heatmaps • Gemini AI coaching • <strong className="text-blue-300">{activeExercise.label}</strong></p>
          </div>
        )}
      </div>
    </section>
  )
}
