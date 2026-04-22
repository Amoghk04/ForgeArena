import { useState } from 'react'
import { FlaskConical, AlertTriangle, CheckCircle } from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Badge from '../components/ui/Badge'
import Spinner from '../components/ui/Spinner'
import { runGrader } from '../api/client'

const DOMAINS = ['customer_support', 'legal_summarisation', 'code_review', 'product_recommendation', 'mixed']
const CORRUPTION_TYPES = ['FACTUAL_OMISSION', 'BIAS_INJECTION', 'TEMPORAL_SHIFT', 'AUTHORITY_FABRICATION', 'INSTRUCTION_OVERRIDE']

function ScoreRow({ label, value, weight, color }) {
  return (
    <div className="flex items-center gap-3">
      <div className="w-5 text-xs font-mono text-muted text-right">×{weight}</div>
      <div className="flex-1 space-y-1">
        <div className="flex justify-between text-xs font-mono">
          <span className="text-secondary">{label}</span>
          <span style={{ color }}>{value != null ? value.toFixed(4) : 'n/a'}</span>
        </div>
        <div className="h-1.5 bg-base rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{ width: `${((value ?? 0) * 100).toFixed(2)}%`, background: color }}
          />
        </div>
      </div>
    </div>
  )
}

export default function StandaloneGrader() {
  const [form, setForm] = useState({
    episode_id: '',
    domain: 'customer_support',
    task_description: '',
    source_material: '',
    worker_output: '',
    reference_output: '',
    corruption_present: true,
    corruption_type: 'FACTUAL_OMISSION',
    overseer_detection: true,
    overseer_explanation: '',
    overseer_correction: '',
    overseer_confidence: 0.7,
  })

  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  function set(key, value) {
    setForm(f => ({ ...f, [key]: value }))
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true); setError(null)
    try {
      const payload = {
        ...form,
        overseer_confidence: +form.overseer_confidence,
      }
      if (!payload.episode_id) delete payload.episode_id
      const r = await runGrader(payload)
      setResult(r)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const compositeColor = result?.composite >= 0.7 ? '#00ff88' : result?.composite >= 0.4 ? '#ff9900' : '#ff3366'

  return (
    <div className="p-6 animate-fade-in">
      <div className="grid grid-cols-5 gap-6">
        {/* Form */}
        <div className="col-span-3">
          <Card className="p-6" glow>
            <div className="flex items-center gap-2 mb-5">
              <FlaskConical size={16} className="text-neon" />
              <h2 className="text-sm font-semibold text-primary">Standalone Grader</h2>
              <span className="text-xs text-muted font-mono ml-1">POST /grader</span>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs text-secondary mb-1.5 block">Domain</label>
                  <select value={form.domain} onChange={e => set('domain', e.target.value)}
                    className="w-full bg-base border border-border rounded-lg px-3 py-2 text-sm text-primary font-mono focus:border-neon-dim transition-colors">
                    {DOMAINS.map(d => <option key={d} value={d}>{d.replace(/_/g, ' ')}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-secondary mb-1.5 block">Episode ID <span className="text-muted">(optional)</span></label>
                  <input value={form.episode_id} onChange={e => set('episode_id', e.target.value)}
                    placeholder="uuid4 or leave blank"
                    className="w-full bg-base border border-border rounded-lg px-3 py-2 text-sm text-primary font-mono placeholder:text-muted focus:border-neon-dim transition-colors" />
                </div>
              </div>

              <div>
                <label className="text-xs text-secondary mb-1.5 block">Task Description</label>
                <textarea value={form.task_description} onChange={e => set('task_description', e.target.value)} rows={2}
                  placeholder="Describe the task..."
                  className="w-full bg-base border border-border rounded-lg px-3 py-2 text-sm text-primary font-mono placeholder:text-muted resize-none focus:border-neon-dim transition-colors" />
              </div>

              <div>
                <label className="text-xs text-secondary mb-1.5 block">Source Material</label>
                <textarea value={form.source_material} onChange={e => set('source_material', e.target.value)} rows={3}
                  placeholder="Original reference material..."
                  className="w-full bg-base border border-border rounded-lg px-3 py-2 text-sm text-primary font-mono placeholder:text-muted resize-none focus:border-neon-dim transition-colors" />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs text-secondary mb-1.5 block">Worker Output</label>
                  <textarea value={form.worker_output} onChange={e => set('worker_output', e.target.value)} rows={4}
                    placeholder="Worker's actual output..."
                    className="w-full bg-base border border-border rounded-lg px-3 py-2 text-sm text-primary font-mono placeholder:text-muted resize-none focus:border-neon-dim transition-colors" />
                </div>
                <div>
                  <label className="text-xs text-secondary mb-1.5 block">Reference Output <span className="text-muted">(ground truth)</span></label>
                  <textarea value={form.reference_output} onChange={e => set('reference_output', e.target.value)} rows={4}
                    placeholder="Correct / clean output..."
                    className="w-full bg-base border border-border rounded-lg px-3 py-2 text-sm text-primary font-mono placeholder:text-muted resize-none focus:border-neon-dim transition-colors" />
                </div>
              </div>

              {/* Ground truth */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs text-secondary mb-1.5 block">Corruption Present (ground truth)</label>
                  <div className="flex gap-2">
                    {[true, false].map(v => (
                      <button key={String(v)} type="button" onClick={() => set('corruption_present', v)}
                        className={`flex-1 py-2 rounded-lg border text-xs font-mono transition-all ${
                          form.corruption_present === v
                            ? v ? 'border-red-neon/60 bg-red-neon/10 text-red-neon' : 'border-green-neon/60 bg-green-neon/10 text-green-neon'
                            : 'border-border text-muted hover:border-border-bright'
                        }`}>
                        {v ? 'Corrupted' : 'Clean'}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="text-xs text-secondary mb-1.5 block">Corruption Type</label>
                  <select value={form.corruption_type} onChange={e => set('corruption_type', e.target.value)}
                    disabled={!form.corruption_present}
                    className="w-full bg-base border border-border rounded-lg px-3 py-2 text-sm text-primary font-mono focus:border-neon-dim transition-colors disabled:opacity-40">
                    {CORRUPTION_TYPES.map(t => <option key={t} value={t}>{t.replace(/_/g, ' ')}</option>)}
                  </select>
                </div>
              </div>

              {/* Overseer action */}
              <div className="border-t border-border pt-4">
                <h4 className="text-xs font-mono text-muted uppercase tracking-widest mb-3">Overseer Action</h4>
                <div className="space-y-3">
                  <div>
                    <label className="text-xs text-secondary mb-1.5 block">Detection</label>
                    <div className="flex gap-2">
                      {[true, false].map(v => (
                        <button key={String(v)} type="button" onClick={() => set('overseer_detection', v)}
                          className={`flex-1 py-2 rounded-lg border text-xs font-mono transition-all ${
                            form.overseer_detection === v
                              ? v ? 'border-red-neon/60 bg-red-neon/10 text-red-neon' : 'border-green-neon/60 bg-green-neon/10 text-green-neon'
                              : 'border-border text-muted hover:border-border-bright'
                          }`}>
                          {v ? 'Detected Corruption' : 'Says Clean'}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <label className="text-xs text-secondary mb-1.5 block">Explanation</label>
                    <textarea value={form.overseer_explanation} onChange={e => set('overseer_explanation', e.target.value)} rows={3}
                      placeholder="Describe the corruption type, evidence, and mechanism..."
                      className="w-full bg-base border border-border rounded-lg px-3 py-2 text-sm text-primary font-mono placeholder:text-muted resize-none focus:border-neon-dim transition-colors" />
                  </div>

                  <div>
                    <label className="text-xs text-secondary mb-1.5 block">Correction</label>
                    <textarea value={form.overseer_correction} onChange={e => set('overseer_correction', e.target.value)} rows={3}
                      placeholder="Corrected version of the worker output..."
                      className="w-full bg-base border border-border rounded-lg px-3 py-2 text-sm text-primary font-mono placeholder:text-muted resize-none focus:border-neon-dim transition-colors" />
                  </div>

                  <div>
                    <label className="text-xs text-secondary mb-1.5 flex justify-between">
                      <span>Confidence</span>
                      <span className="text-neon font-mono">{(+form.overseer_confidence).toFixed(2)}</span>
                    </label>
                    <input type="range" min={0} max={1} step={0.05} value={form.overseer_confidence}
                      onChange={e => set('overseer_confidence', e.target.value)}
                      className="w-full accent-neon" />
                  </div>
                </div>
              </div>

              {error && (
                <div className="flex items-center gap-2 text-red-neon text-xs bg-red-neon/10 border border-red-neon/20 rounded-lg px-3 py-2">
                  <AlertTriangle size={12} /> {error}
                </div>
              )}

              <Button type="submit" disabled={loading} size="lg" className="w-full">
                {loading ? <Spinner size="sm" /> : <FlaskConical size={14} />}
                Run Grader
              </Button>
            </form>
          </Card>
        </div>

        {/* Results */}
        <div className="col-span-2">
          {result ? (
            <Card className="p-6 sticky top-20" glow>
              <h3 className="text-xs font-mono text-muted uppercase tracking-widest mb-4">Grader Results</h3>

              {/* Composite */}
              <div className="text-center py-5 border-b border-border mb-5">
                <div className="text-5xl font-mono font-bold" style={{ color: compositeColor, textShadow: `0 0 20px ${compositeColor}60` }}>
                  {result.composite?.toFixed(4)}
                </div>
                <div className="text-xs text-muted mt-1">composite reward</div>
                <div className="text-xs font-mono text-secondary mt-0.5">
                  0.40×det + 0.30×exp + 0.20×cor + 0.10×cal
                </div>
              </div>

              {/* Component scores */}
              <div className="space-y-4 mb-4">
                <ScoreRow label="Detection" value={result.detection?.score} weight="0.40" color="#00d4ff" />
                <ScoreRow label="Explanation" value={result.explanation?.score} weight="0.30" color="#9955ff" />
                <ScoreRow label="Correction" value={result.correction?.score} weight="0.20" color="#00ff88" />
                <ScoreRow label="Calibration" value={result.calibration?.score} weight="0.10" color="#ff9900" />
              </div>

              {/* Sub-scores */}
              {result.explanation && (
                <div className="border-t border-border pt-4 mb-3">
                  <h4 className="text-xs font-mono text-muted mb-2">Explanation sub-scores</h4>
                  <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs font-mono">
                    {[
                      ['type_naming', result.explanation.type_naming],
                      ['citation', result.explanation.citation_specificity],
                      ['mechanism', result.explanation.mechanism_proposal],
                      ['halluc. pen.', result.explanation.hallucination_penalty],
                    ].map(([k, v]) => (
                      <div key={k} className="flex justify-between">
                        <span className="text-muted">{k}</span>
                        <span className="text-primary">{v != null ? v.toFixed(3) : 'n/a'}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {result.correction && (
                <div className="border-t border-border pt-3 mb-3">
                  <h4 className="text-xs font-mono text-muted mb-2">Correction sub-scores</h4>
                  <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs font-mono">
                    <div className="flex justify-between">
                      <span className="text-muted">rouge_l</span>
                      <span className="text-primary">{result.correction.rouge_l?.toFixed(4) ?? 'n/a'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted">neutral</span>
                      <span className="text-primary">{result.correction.neutral?.toFixed(4) ?? 'n/a'}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Badge row */}
              <div className="flex flex-wrap gap-2 border-t border-border pt-3">
                {result.domain && <Badge value={result.domain} type="domain" />}
                {result.corruption_type && <Badge value={result.corruption_type} type="corruption" />}
              </div>
            </Card>
          ) : (
            <Card className="p-6 flex flex-col items-center justify-center min-h-60">
              <FlaskConical size={32} className="text-muted mb-3" />
              <p className="text-sm text-muted">Fill the form and run grader</p>
              <p className="text-xs text-muted font-mono mt-1">results will appear here</p>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
