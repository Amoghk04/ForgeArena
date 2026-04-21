import { useState, useCallback, useMemo } from 'react'
import {
  ScatterChart, Scatter, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, Cell,
} from 'recharts'
import { Search, Filter, RefreshCw } from 'lucide-react'
import Card from '../components/ui/Card'
import Badge from '../components/ui/Badge'
import Button from '../components/ui/Button'
import Spinner from '../components/ui/Spinner'
import { usePolling } from '../hooks/usePolling'
import { getTasks } from '../api/client'

const TIER_COLOR = {
  learnable: '#00ff88',
  'too-easy': '#00d4ff',
  'too-hard': '#ff3366',
  unestimated: '#2e3452',
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  if (!d) return null
  return (
    <div className="custom-tooltip text-xs space-y-1">
      <p className="text-primary font-semibold">{d.task_id?.slice(0, 16)}</p>
      <p><span className="text-muted">domain: </span><span className="text-neon">{d.domain}</span></p>
      <p><span className="text-muted">tier: </span><span style={{ color: TIER_COLOR[d.difficulty_tier] }}>{d.difficulty_tier}</span></p>
      <p><span className="text-muted">pass@k: </span><span className="text-primary">{d.pass_at_k != null ? d.pass_at_k.toFixed(3) : 'n/a'}</span></p>
      <p><span className="text-muted">sophistication: </span><span className="text-primary">{d.corruption_sophistication}</span></p>
      <p><span className="text-muted">obfuscation: </span><span className="text-primary">{d.obfuscation_depth}</span></p>
    </div>
  )
}

const OBF_ORDER = { low: 1, medium: 2, high: 3 }

export default function TaskBank() {
  const fetcher = useCallback(() => getTasks(), [])
  const { data, loading, error, refresh } = usePolling(fetcher, 30000)

  const [search, setSearch] = useState('')
  const [filterDomain, setFilterDomain] = useState('all')
  const [filterTier, setFilterTier] = useState('all')

  const tasks = data?.tasks ?? []

  const domains = useMemo(() => ['all', ...new Set(tasks.map(t => t.domain))], [tasks])
  const tiers = ['all', 'learnable', 'too-easy', 'too-hard', 'unestimated']

  const filtered = useMemo(() => tasks.filter(t => {
    if (filterDomain !== 'all' && t.domain !== filterDomain) return false
    if (filterTier !== 'all' && t.difficulty_tier !== filterTier) return false
    if (search && !t.task_id?.includes(search) && !t.domain?.includes(search) && !t.task_description?.toLowerCase().includes(search.toLowerCase())) return false
    return true
  }), [tasks, filterDomain, filterTier, search])

  // Scatter data: x = corruption_sophistication, y = obfuscation (encoded), colored by tier
  const scatterData = useMemo(() =>
    filtered.map(t => ({
      ...t,
      x: t.corruption_sophistication ?? 0,
      y: OBF_ORDER[t.obfuscation_depth] ?? 2,
    })), [filtered])

  const stats = useMemo(() => ({
    total: tasks.length,
    learnable: tasks.filter(t => t.difficulty_tier === 'learnable').length,
    tooEasy: tasks.filter(t => t.difficulty_tier === 'too-easy').length,
    tooHard: tasks.filter(t => t.difficulty_tier === 'too-hard').length,
    generated: tasks.filter(t => t.is_generated).length,
  }), [tasks])

  return (
    <div className="p-6 space-y-5 animate-fade-in">
      {/* Stats strip */}
      <div className="grid grid-cols-5 gap-3">
        {[
          ['Total Tasks', stats.total, '#7080aa'],
          ['Learnable', stats.learnable, '#00ff88'],
          ['Too Easy', stats.tooEasy, '#00d4ff'],
          ['Too Hard', stats.tooHard, '#ff3366'],
          ['Generated', stats.generated, '#9955ff'],
        ].map(([label, val, color]) => (
          <Card key={label} className="p-4" glow>
            <div className="text-2xl font-mono font-bold" style={{ color }}>{val}</div>
            <div className="text-xs text-muted mt-0.5">{label}</div>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-5 gap-5">
        {/* Table */}
        <div className="col-span-3 space-y-3">
          {/* Filters */}
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
              <input
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Search tasks…"
                className="w-full bg-card border border-border rounded-lg pl-9 pr-3 py-2 text-sm text-primary font-mono placeholder:text-muted focus:border-neon-dim transition-colors"
              />
            </div>
            <select
              value={filterDomain}
              onChange={e => setFilterDomain(e.target.value)}
              className="bg-card border border-border text-sm text-secondary rounded-lg px-3 py-2 font-mono focus:border-neon-dim transition-colors"
            >
              {domains.map(d => <option key={d} value={d}>{d === 'all' ? 'All Domains' : d.replace(/_/g, ' ')}</option>)}
            </select>
            <select
              value={filterTier}
              onChange={e => setFilterTier(e.target.value)}
              className="bg-card border border-border text-sm text-secondary rounded-lg px-3 py-2 font-mono focus:border-neon-dim transition-colors"
            >
              {tiers.map(t => <option key={t} value={t}>{t === 'all' ? 'All Tiers' : t}</option>)}
            </select>
            <Button onClick={refresh} variant="secondary" size="md">
              <RefreshCw size={13} />
            </Button>
          </div>

          {/* Table */}
          <Card>
            <div className="overflow-auto max-h-[540px]">
              {loading && !data ? (
                <div className="flex justify-center py-12"><Spinner /></div>
              ) : error ? (
                <div className="text-center py-12 text-red-neon text-sm">{error}</div>
              ) : (
                <table className="w-full text-xs font-mono">
                  <thead className="sticky top-0 bg-card">
                    <tr className="border-b border-border text-muted">
                      <th className="text-left py-3 px-4">ID</th>
                      <th className="text-left py-3 px-4">Domain</th>
                      <th className="text-left py-3 px-4">Tier</th>
                      <th className="text-right py-3 px-4">pass@k</th>
                      <th className="text-right py-3 px-4">Soph.</th>
                      <th className="text-left py-3 px-4">Obfusc.</th>
                      <th className="text-left py-3 px-4">Gen?</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map(t => (
                      <tr key={t.task_id} className="border-b border-border/40 hover:bg-card-hover transition-colors">
                        <td className="py-2.5 px-4 text-muted">{t.task_id?.slice(0, 10)}…</td>
                        <td className="py-2.5 px-4"><Badge value={t.domain} type="domain" /></td>
                        <td className="py-2.5 px-4"><Badge value={t.difficulty_tier ?? 'unestimated'} type="tier" /></td>
                        <td className="py-2.5 px-4 text-right" style={{ color: TIER_COLOR[t.difficulty_tier] || '#7080aa' }}>
                          {t.pass_at_k != null ? t.pass_at_k.toFixed(3) : '—'}
                        </td>
                        <td className="py-2.5 px-4 text-right text-primary">{t.corruption_sophistication ?? '—'}</td>
                        <td className="py-2.5 px-4">
                          <span className={`px-1.5 py-0.5 rounded text-xs ${
                            t.obfuscation_depth === 'high' ? 'text-red-neon bg-red-neon/10' :
                            t.obfuscation_depth === 'medium' ? 'text-orange-neon bg-orange-neon/10' :
                            'text-green-neon bg-green-neon/10'
                          }`}>{t.obfuscation_depth ?? '—'}</span>
                        </td>
                        <td className="py-2.5 px-4">
                          {t.is_generated ? <span className="text-purple-neon">gen</span> : <span className="text-muted">seed</span>}
                        </td>
                      </tr>
                    ))}
                    {filtered.length === 0 && (
                      <tr><td colSpan={7} className="py-12 text-center text-muted">no tasks match filter</td></tr>
                    )}
                  </tbody>
                </table>
              )}
            </div>
          </Card>
        </div>

        {/* Scatter */}
        <div className="col-span-2">
          <Card className="p-5 h-full" glow>
            <h3 className="text-sm font-semibold text-secondary mb-1">Sophistication vs Obfuscation</h3>
            <p className="text-xs text-muted mb-4">Coloured by difficulty tier</p>
            {scatterData.length > 0 ? (
              <>
                <ResponsiveContainer width="100%" height={340}>
                  <ScatterChart margin={{ left: -10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1c1c3a" />
                    <XAxis dataKey="x" name="Sophistication"
                      label={{ value: 'Sophistication', position: 'insideBottom', offset: -4, fill: '#7080aa', fontSize: 10 }}
                      tick={{ fill: '#7080aa', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                    />
                    <YAxis dataKey="y" name="Obfuscation" domain={[0, 4]}
                      tickFormatter={v => ['', 'low', 'med', 'high'][v] || ''}
                      tick={{ fill: '#7080aa', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                    />
                    <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(0,212,255,0.05)' }} />
                    <Scatter data={scatterData} isAnimationActive={false}>
                      {scatterData.map((d, i) => (
                        <Cell key={i} fill={TIER_COLOR[d.difficulty_tier] || '#2e3452'}
                          style={{ filter: `drop-shadow(0 0 4px ${TIER_COLOR[d.difficulty_tier] || '#2e3452'}80)` }}
                        />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
                <div className="flex flex-wrap gap-3 mt-3">
                  {Object.entries(TIER_COLOR).map(([tier, color]) => (
                    <span key={tier} className="flex items-center gap-1 text-xs font-mono text-secondary">
                      <span className="w-2 h-2 rounded-full" style={{ background: color }} />
                      {tier}
                    </span>
                  ))}
                </div>
              </>
            ) : (
              <div className="flex items-center justify-center h-64 text-muted text-sm">no tasks</div>
            )}
          </Card>
        </div>
      </div>
    </div>
  )
}
