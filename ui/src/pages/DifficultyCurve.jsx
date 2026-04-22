import { useState, useCallback, useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts'
import { RefreshCw } from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Spinner from '../components/ui/Spinner'
import Badge from '../components/ui/Badge'
import { usePolling } from '../hooks/usePolling'
import { getDifficultyCurve } from '../api/client'

const PALETTE = [
  '#00d4ff', '#00ff88', '#9955ff', '#ff9900', '#ff3366',
  '#ffee00', '#00ffcc', '#ff66bb', '#88aaff', '#ff7744',
]

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="custom-tooltip max-w-[220px]">
      <p className="text-muted mb-2 text-[10px]">Snapshot {label}</p>
      {payload.slice(0, 6).map(p => (
        <p key={p.name} style={{ color: p.color }} className="text-[10px] truncate">
          {p.name.slice(0, 18)}: <b>{typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</b>
        </p>
      ))}
      {payload.length > 6 && <p className="text-muted text-[9px]">+{payload.length - 6} more</p>}
    </div>
  )
}

export default function DifficultyCurve() {
  const fetcher = useCallback(() => getDifficultyCurve(), [])
  const { data, loading, error, refresh } = usePolling(fetcher, 8000)

  const [selectedTasks, setSelectedTasks] = useState(new Set())

  const taskIds = useMemo(() => Object.keys(data ?? {}), [data])

  // Build chart data: array of {snapshot: n, task_id: pass_at_k}
  const chartData = useMemo(() => {
    if (!data) return []
    const maxLen = Math.max(...Object.values(data).map(arr => arr.length), 0)
    return Array.from({ length: maxLen }, (_, i) => {
      const row = { snapshot: i + 1 }
      Object.entries(data).forEach(([tid, snaps]) => {
        if (snaps[i] != null) {
          const snap = typeof snaps[i] === 'object' ? (snaps[i].pass_at_k ?? snaps[i]) : snaps[i]
          row[tid] = +snap.toFixed(4)
        }
      })
      return row
    })
  }, [data])

  const visibleTasks = selectedTasks.size === 0 ? taskIds.slice(0, 10) : Array.from(selectedTasks)

  function toggleTask(tid) {
    setSelectedTasks(prev => {
      const next = new Set(prev)
      if (next.has(tid)) next.delete(tid)
      else next.add(tid)
      return next
    })
  }

  return (
    <div className="p-6 space-y-5 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-primary">Difficulty Curve</h2>
          <p className="text-xs text-muted font-mono mt-0.5">pass@k time series - primary Forge visualisation</p>
        </div>
        <Button onClick={refresh} variant="secondary" size="sm">
          <RefreshCw size={13} /> refresh
        </Button>
      </div>

      {/* Chart */}
      <Card className="p-5" glow>
        <h3 className="text-sm font-semibold text-secondary mb-1">pass@k per Task</h3>
        <p className="text-xs text-muted mb-4">Learnable zone: 0.20 – 0.85 (between dashed lines)</p>
        {loading && !data ? (
          <div className="flex items-center justify-center h-72"><Spinner /></div>
        ) : chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={chartData} margin={{ left: -10, right: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1c1c3a" />
              <XAxis
                dataKey="snapshot"
                tick={{ fill: '#7080aa', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                label={{ value: 'Snapshot', position: 'insideBottom', offset: -4, fill: '#7080aa', fontSize: 10 }}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fill: '#7080aa', fontSize: 10, fontFamily: 'JetBrains Mono' }}
              />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={0.20} stroke="#ff3366" strokeDasharray="4 4" strokeWidth={1.5}
                label={{ value: '0.20 (too-hard)', fill: '#ff3366', fontSize: 9, fontFamily: 'JetBrains Mono', position: 'insideTopLeft' }} />
              <ReferenceLine y={0.85} stroke="#00d4ff" strokeDasharray="4 4" strokeWidth={1.5}
                label={{ value: '0.85 (too-easy)', fill: '#00d4ff', fontSize: 9, fontFamily: 'JetBrains Mono', position: 'insideTopLeft' }} />
              {visibleTasks.map((tid, i) => (
                <Line
                  key={tid}
                  type="monotone"
                  dataKey={tid}
                  name={tid.slice(0, 20)}
                  stroke={PALETTE[i % PALETTE.length]}
                  strokeWidth={1.5}
                  dot={false}
                  activeDot={{ r: 4, style: { filter: `drop-shadow(0 0 4px ${PALETTE[i % PALETTE.length]})` } }}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex items-center justify-center h-72 text-muted text-sm">
            {error ? <span className="text-red-neon">{error}</span> : 'No difficulty data yet. Run episodes to populate.'}
          </div>
        )}
      </Card>

      {/* Task selector */}
      {taskIds.length > 0 && (
        <Card className="p-5">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-secondary">Task Selector</h3>
            <div className="flex gap-2">
              <Button onClick={() => setSelectedTasks(new Set())} variant="ghost" size="sm">show first 10</Button>
              <Button onClick={() => setSelectedTasks(new Set(taskIds))} variant="secondary" size="sm">show all</Button>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {taskIds.map((tid, i) => {
              const selected = selectedTasks.size === 0 ? i < 10 : selectedTasks.has(tid)
              return (
                <button
                  key={tid}
                  onClick={() => toggleTask(tid)}
                  className={`px-2.5 py-1 rounded text-xs font-mono border transition-all ${
                    selected
                      ? 'border-opacity-60 text-opacity-100'
                      : 'border-border text-muted hover:border-border-bright'
                  }`}
                  style={selected ? {
                    borderColor: PALETTE[i % PALETTE.length],
                    color: PALETTE[i % PALETTE.length],
                    background: `${PALETTE[i % PALETTE.length]}10`,
                  } : {}}
                >
                  {tid.slice(0, 14)}
                </button>
              )
            })}
          </div>
          {taskIds.length > 10 && selectedTasks.size === 0 && (
            <p className="text-xs text-muted font-mono mt-2">{taskIds.length - 10} more tasks not shown. Click a task to filter.</p>
          )}
        </Card>
      )}

      {/* Stats per shown task */}
      {data && visibleTasks.length > 0 && (
        <Card className="p-5">
          <h3 className="text-sm font-semibold text-secondary mb-3">Snapshot Summary</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="border-b border-border text-muted">
                  <th className="text-left py-2 pr-4">Task ID</th>
                  <th className="text-right py-2 pr-4">Snapshots</th>
                  <th className="text-right py-2 pr-4">Latest pass@k</th>
                  <th className="text-right py-2 pr-4">Min</th>
                  <th className="text-right py-2">Max</th>
                </tr>
              </thead>
              <tbody>
                {visibleTasks.map((tid, i) => {
                  const snaps = data[tid] ?? []
                  const vals = snaps.map(s => typeof s === 'object' ? (s.pass_at_k ?? 0) : s)
                  const latest = vals[vals.length - 1]
                  const min = Math.min(...vals)
                  const max = Math.max(...vals)
                  const tier = latest < 0.20 ? 'too-hard' : latest > 0.85 ? 'too-easy' : 'learnable'
                  return (
                    <tr key={tid} className="border-b border-border/40 hover:bg-card-hover transition-colors">
                      <td className="py-2 pr-4" style={{ color: PALETTE[i % PALETTE.length] }}>{tid.slice(0, 18)}</td>
                      <td className="py-2 pr-4 text-right text-muted">{snaps.length}</td>
                      <td className="py-2 pr-4 text-right"><Badge value={tier} type="tier" /></td>
                      <td className="py-2 pr-4 text-right text-muted">{min.toFixed(3)}</td>
                      <td className="py-2 text-right text-muted">{max.toFixed(3)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}
