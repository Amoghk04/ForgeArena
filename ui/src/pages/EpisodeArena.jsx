import { useState } from 'react'
import {
  Play, MessageSquare, Eye, CheckCircle, XCircle, RefreshCw,
  ChevronRight, AlertTriangle, Info, Loader2,
} from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Badge from '../components/ui/Badge'
import Spinner from '../components/ui/Spinner'
import { resetEpisode, stepEpisode, getState } from '../api/client'

const DOMAINS = [
  'customer_support',
  'legal_summarisation',
  'code_review',
  'product_recommendation',
  'mixed',
]

const PHASE_STEPS = ['Start Episode', 'Inspect', 'Results']

function ScoreBar({ label, value, color = '#00d4ff', width = '100%' }) {
  return (
    <div className="space-y-1" style={{ width }}>
      <div className="flex justify-between text-xs font-mono">
        <span className="text-secondary">{label}</span>
        <span style={{ color }}>{value != null ? value.toFixed(3) : '—'}</span>
      </div>
      <div className="h-1.5 bg-base rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${((value ?? 0) * 100).toFixed(1)}%`, background: color, boxShadow: `0 0 6px ${color}80` }}
        />
      </div>
    </div>
  )
}

function PhaseIndicator({ current }) {
  return (
    <div className="flex items-center gap-2 mb-6">
      {PHASE_STEPS.map((step, i) => {
        const done = i < current
        const active = i === current
        return (
          <div key={step} className="flex items-center gap-2">
            <div className={`flex items-center gap-1.5 text-xs font-mono px-3 py-1.5 rounded-full border transition-all ${
              active ? 'border-neon text-neon bg-neon-dark' :
              done ? 'border-green-neon/30 text-green-neon bg-green-neon/5' :
              'border-border text-muted'
            }`}>
              {done ? <CheckCircle size={11} /> : <span className="w-3 h-3 rounded-full border border-current flex items-center justify-center text-[9px]">{i+1}</span>}
              {step}
            </div>
            {i < PHASE_STEPS.length - 1 && <ChevronRight size={12} className="text-muted" />}
          </div>
        )
      })}
    </div>
  )
}

export default function EpisodeArena() {
  const [domain, setDomain] = useState('customer_support')
  const [phase, setPhase] = useState(0) // 0=idle, 1=inspecting, 2=done
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Episode state
  const [reset, setReset] = useState(null)       // ResetObservation
  const [probeQuestion, setProbeQuestion] = useState('')
  const [probeResult, setProbeResult] = useState(null)  // WorkerObservation
  const [probeLoading, setProbeLoading] = useState(false)
  const [probeUsed, setProbeUsed] = useState(false)

  // Inspect form
  const [detection, setDetection] = useState(true)
  const [explanation, setExplanation] = useState('')
  const [correction, setCorrection] = useState('')
  const [confidence, setConfidence] = useState(0.7)

  // Results
  const [result, setResult] = useState(null)   // EpisodeResult

  // State panel
  const [stateData, setStateData] = useState(null)
  const [stateLoading, setStateLoading] = useState(false)

  // History
  const [history, setHistory] = useState([])

  async function handleReset() {
    setLoading(true); setError(null)
    try {
      const obs = await resetEpisode(domain)
      setReset(obs)
      setProbeResult(null)
      setProbeQuestion('')
      setProbeUsed(false)
      setDetection(true)
      setExplanation('')
      setCorrection('')
      setConfidence(0.7)
      setResult(null)
      setStateData(null)
      setPhase(1)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleProbe() {
    if (!probeQuestion.trim() || !reset) return
    setProbeLoading(true); setError(null)
    try {
      const obs = await stepEpisode(reset.episode_id, {
        action_type: 'overseer_probe',
        question: probeQuestion,
      })
      setProbeResult(obs)
      setProbeUsed(true)
    } catch (e) {
      setError(e.message)
    } finally {
      setProbeLoading(false)
    }
  }

  async function handleInspect() {
    if (!reset) return
    setLoading(true); setError(null)
    try {
      const res = await stepEpisode(reset.episode_id, {
        action_type: 'overseer_inspect',
        detection,
        explanation,
        correction,
        confidence,
      })
      setResult(res)
      setPhase(2)
      setHistory(h => [{ episode_id: reset.episode_id, domain: reset.domain, result: res }, ...h.slice(0, 9)])
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleGetState() {
    if (!reset?.episode_id) return
    setStateLoading(true)
    try {
      const s = await getState(reset.episode_id)
      setStateData(s)
    } catch (e) {
      setStateData({ error: e.message })
    } finally {
      setStateLoading(false)
    }
  }

  const corrCorrect = result?.corruption_present != null && result?.corruption_present === detection
  const reward = result?.reward

  return (
    <div className="p-6 space-y-5 animate-fade-in">
      <PhaseIndicator current={phase} />

      <div className="grid grid-cols-5 gap-5">
        {/* Left: controls */}
        <div className="col-span-2 space-y-4">
          {/* Reset panel */}
          <Card className="p-5" glow>
            <h3 className="text-xs font-mono text-muted uppercase tracking-widest mb-4">Phase 1 — Start Episode</h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-secondary mb-1.5 block">Domain</label>
                <select
                  value={domain}
                  onChange={e => setDomain(e.target.value)}
                  className="w-full bg-base border border-border text-primary text-sm rounded-lg px-3 py-2 font-mono focus:border-neon-dim transition-colors"
                >
                  {DOMAINS.map(d => (
                    <option key={d} value={d}>{d.replace(/_/g, ' ')}</option>
                  ))}
                </select>
              </div>
              <Button onClick={handleReset} disabled={loading || phase === 1 || phase === 2} size="lg" className="w-full">
                {loading && phase === 0 ? <Spinner size="sm" /> : <Play size={14} />}
                {phase === 0 ? 'Start New Episode' : phase === 1 ? 'Episode Active' : 'Episode Done'}
              </Button>
              {(phase === 1 || phase === 2) && (
                <Button onClick={() => { setPhase(0); setReset(null); setResult(null) }} variant="secondary" size="sm" className="w-full">
                  <RefreshCw size={12} /> New Episode
                </Button>
              )}
            </div>
          </Card>

          {/* Probe panel */}
          {phase === 1 && (
            <Card className="p-5">
              <h3 className="text-xs font-mono text-muted uppercase tracking-widest mb-1">Phase 2 — Probe Worker <span className="text-muted">(optional, once)</span></h3>
              <p className="text-xs text-muted mb-3">Ask the Worker a clarifying question about its response.</p>
              {probeUsed ? (
                <div className="flex items-center gap-2 text-xs text-muted bg-base rounded-lg px-3 py-2 border border-border">
                  <Info size={12} /> Probe already used this episode
                </div>
              ) : (
                <div className="space-y-2">
                  <textarea
                    value={probeQuestion}
                    onChange={e => setProbeQuestion(e.target.value)}
                    placeholder="e.g. What sources did you cite for the pricing data?"
                    rows={3}
                    className="w-full bg-base border border-border rounded-lg px-3 py-2 text-sm text-primary font-mono placeholder:text-muted resize-none focus:border-neon-dim transition-colors"
                  />
                  <Button onClick={handleProbe} disabled={probeLoading || !probeQuestion.trim()} size="sm" className="w-full">
                    {probeLoading ? <Spinner size="sm" /> : <MessageSquare size={12} />}
                    Send Probe
                  </Button>
                </div>
              )}
            </Card>
          )}

          {/* State inspector */}
          {phase >= 1 && (
            <Card className="p-4">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-mono text-muted uppercase tracking-widest">GET /state</h3>
                <Button onClick={handleGetState} variant="ghost" size="sm" disabled={stateLoading}>
                  {stateLoading ? <Spinner size="sm" /> : <Eye size={12} />} inspect
                </Button>
              </div>
              {stateData && (
                <pre className="text-xs font-mono text-secondary bg-base rounded p-2 max-h-32 overflow-auto">
                  {JSON.stringify(stateData, null, 2)}
                </pre>
              )}
            </Card>
          )}
        </div>

        {/* Right: main content */}
        <div className="col-span-3 space-y-4">
          {/* Task display */}
          {reset && (
            <Card className="p-5" glow>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Badge value={reset.domain} type="domain" />
                  <span className="text-xs font-mono text-muted">{reset.episode_id?.slice(0, 8)}…</span>
                </div>
                <span className="text-xs font-mono text-muted">OVERSEER_INSPECTING</span>
              </div>
              <h4 className="text-sm font-semibold text-primary mb-2">Task Description</h4>
              <p className="text-sm text-secondary leading-relaxed mb-4">{reset.task_description}</p>
              {reset.source_material && (
                <>
                  <h4 className="text-xs font-mono text-muted uppercase tracking-widest mb-2">Source Material</h4>
                  <div className="text-xs text-secondary font-mono bg-base rounded-lg p-3 border border-border max-h-32 overflow-auto leading-relaxed">
                    {reset.source_material}
                  </div>
                </>
              )}
            </Card>
          )}

          {/* Probe result */}
          {probeResult && (
            <Card className="p-5">
              <h4 className="text-xs font-mono text-neon uppercase tracking-widest mb-3">Worker Response (via Probe)</h4>
              {probeResult.worker_cot && (
                <div className="mb-3">
                  <p className="text-xs font-mono text-muted uppercase tracking-widest mb-1">Chain of Thought</p>
                  <div className="text-xs text-secondary font-mono bg-base rounded p-3 border border-border max-h-32 overflow-auto leading-relaxed whitespace-pre-wrap">
                    {probeResult.worker_cot}
                  </div>
                </div>
              )}
              {probeResult.worker_output && (
                <div>
                  <p className="text-xs font-mono text-muted uppercase tracking-widest mb-1">Worker Output</p>
                  <div className="text-sm text-primary bg-base rounded p-3 border border-border max-h-40 overflow-auto leading-relaxed whitespace-pre-wrap">
                    {probeResult.worker_output}
                  </div>
                </div>
              )}
            </Card>
          )}

          {/* Inspect form */}
          {phase === 1 && (
            <Card className="p-5">
              <h3 className="text-xs font-mono text-muted uppercase tracking-widest mb-4">Phase 3 — Submit Inspection</h3>
              <div className="space-y-4">
                {/* Detection toggle */}
                <div>
                  <label className="text-xs text-secondary mb-2 block">Corruption Detected?</label>
                  <div className="flex gap-3">
                    {[true, false].map(v => (
                      <button
                        key={String(v)}
                        onClick={() => setDetection(v)}
                        className={`flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-mono transition-all ${
                          detection === v
                            ? v ? 'border-red-neon/60 bg-red-neon/10 text-red-neon' : 'border-green-neon/60 bg-green-neon/10 text-green-neon'
                            : 'border-border text-muted hover:border-border-bright'
                        }`}
                      >
                        {v ? <AlertTriangle size={12} /> : <CheckCircle size={12} />}
                        {v ? 'Corrupted' : 'Clean'}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Explanation */}
                <div>
                  <label className="text-xs text-secondary mb-1.5 block">Explanation <span className="text-muted">(type, citation, mechanism)</span></label>
                  <textarea
                    value={explanation}
                    onChange={e => setExplanation(e.target.value)}
                    placeholder="Describe the corruption type, what evidence you found, and the mechanism..."
                    rows={4}
                    className="w-full bg-base border border-border rounded-lg px-3 py-2 text-sm text-primary font-mono placeholder:text-muted resize-none focus:border-neon-dim transition-colors"
                  />
                </div>

                {/* Correction */}
                <div>
                  <label className="text-xs text-secondary mb-1.5 block">Correction <span className="text-muted">(corrected output)</span></label>
                  <textarea
                    value={correction}
                    onChange={e => setCorrection(e.target.value)}
                    placeholder="Provide the corrected version of the worker's output..."
                    rows={4}
                    className="w-full bg-base border border-border rounded-lg px-3 py-2 text-sm text-primary font-mono placeholder:text-muted resize-none focus:border-neon-dim transition-colors"
                  />
                </div>

                {/* Confidence */}
                <div>
                  <label className="text-xs text-secondary mb-1.5 flex justify-between">
                    <span>Confidence</span>
                    <span className="font-mono text-neon">{confidence.toFixed(2)}</span>
                  </label>
                  <input
                    type="range" min={0} max={1} step={0.05}
                    value={confidence}
                    onChange={e => setConfidence(+e.target.value)}
                    className="w-full accent-neon"
                  />
                  <div className="flex justify-between text-xs text-muted font-mono mt-1">
                    <span>0.00 (uncertain)</span>
                    <span>1.00 (certain)</span>
                  </div>
                </div>

                <Button onClick={handleInspect} disabled={loading || !explanation.trim() || !correction.trim()} size="lg" className="w-full">
                  {loading ? <Spinner size="sm" /> : <Eye size={14} />}
                  Submit Inspection
                </Button>
              </div>
            </Card>
          )}

          {/* Results */}
          {result && phase === 2 && (
            <Card className="p-5" glow>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xs font-mono text-muted uppercase tracking-widest">Episode Results</h3>
                <div className="flex items-center gap-2">
                  {corrCorrect
                    ? <span className="flex items-center gap-1 text-xs text-green-neon font-mono"><CheckCircle size={12}/> Detection Correct</span>
                    : <span className="flex items-center gap-1 text-xs text-red-neon font-mono"><XCircle size={12}/> Detection Wrong</span>
                  }
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 mb-5">
                <div className="bg-base rounded-lg p-3 border border-border">
                  <p className="text-xs text-muted font-mono mb-1">Ground Truth</p>
                  <p className="text-sm font-mono">
                    {result.corruption_present
                      ? <span className="text-red-neon">Corrupted</span>
                      : <span className="text-green-neon">Clean</span>
                    }
                  </p>
                  {result.corruption_type && (
                    <Badge value={result.corruption_type} type="corruption" className="mt-1" />
                  )}
                </div>
                <div className="bg-base rounded-lg p-3 border border-border">
                  <p className="text-xs text-muted font-mono mb-1">Composite Reward</p>
                  <p className="text-3xl font-mono font-bold text-neon">
                    {reward != null ? reward.toFixed(3) : '—'}
                  </p>
                </div>
              </div>

              <div className="space-y-3">
                <h4 className="text-xs font-mono text-muted uppercase tracking-widest">Component Breakdown</h4>
                <ScoreBar label="Detection (×0.40)" value={result.detection_score} color="#00d4ff" />
                <ScoreBar label="Explanation (×0.30)" value={result.explanation_score} color="#9955ff" />
                <ScoreBar label="Correction (×0.20)" value={result.correction_score} color="#00ff88" />
                <ScoreBar label="Calibration (×0.10)" value={result.calibration_score} color="#ff9900" />
              </div>
            </Card>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 text-red-neon text-sm bg-red-neon/10 border border-red-neon/20 rounded-lg px-4 py-3">
          <AlertTriangle size={14} /> {error}
        </div>
      )}

      {/* History */}
      {history.length > 0 && (
        <Card className="p-5">
          <h3 className="text-xs font-mono text-muted uppercase tracking-widest mb-3">Episode History</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="border-b border-border text-muted">
                  <th className="text-left py-2 pr-4">Episode ID</th>
                  <th className="text-left py-2 pr-4">Domain</th>
                  <th className="text-left py-2 pr-4">Corruption</th>
                  <th className="text-right py-2 pr-4">Detection</th>
                  <th className="text-right py-2 pr-4">Explanation</th>
                  <th className="text-right py-2">Composite</th>
                </tr>
              </thead>
              <tbody>
                {history.map(({ episode_id, domain: d, result: r }) => (
                  <tr key={episode_id} className="border-b border-border/50 hover:bg-card-hover transition-colors">
                    <td className="py-2 pr-4 text-muted">{episode_id?.slice(0, 8)}…</td>
                    <td className="py-2 pr-4"><Badge value={d} type="domain" /></td>
                    <td className="py-2 pr-4">
                      {r?.corruption_type
                        ? <Badge value={r.corruption_type} type="corruption" />
                        : <span className="text-muted">—</span>}
                    </td>
                    <td className="py-2 pr-4 text-right text-neon">{r?.detection_score?.toFixed(2) ?? '—'}</td>
                    <td className="py-2 pr-4 text-right text-purple-neon">{r?.explanation_score?.toFixed(2) ?? '—'}</td>
                    <td className="py-2 text-right font-bold text-neon">{r?.reward?.toFixed(3) ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}
