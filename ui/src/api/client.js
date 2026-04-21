const BASE = '/api'

async function request(method, path, body) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  }
  if (body !== undefined) opts.body = JSON.stringify(body)
  const res = await fetch(`${BASE}${path}`, opts)
  if (!res.ok) {
    const err = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${err}`)
  }
  return res.json()
}

// POST /reset — start a new episode
export async function resetEpisode(domain = null, task_id = null) {
  const body = {}
  if (domain) body.domain = domain
  if (task_id) body.task_id = task_id
  return request('POST', '/reset', body)
}

// POST /step — advance episode with an action
export async function stepEpisode(episode_id, action) {
  return request('POST', '/step', { episode_id, action })
}

// GET /state?episode_id= — inspect without advancing
export async function getState(episode_id) {
  return request('GET', `/state?episode_id=${episode_id}`)
}

// GET /tasks — full task bank
export async function getTasks() {
  return request('GET', '/tasks')
}

// POST /grader — standalone offline grader
export async function runGrader(payload) {
  return request('POST', '/grader', payload)
}

// GET /baseline — pre-computed baseline scores
export async function getBaseline() {
  return request('GET', '/baseline')
}

// GET /forge/queue — active queue state
export async function getForgeQueue() {
  return request('GET', '/forge/queue')
}

// GET /forge/stats — aggregate Forge statistics
export async function getForgeStats() {
  return request('GET', '/forge/stats')
}

// GET /oversight/stats — detection/explanation/correction per domain+corruption
export async function getOversightStats() {
  return request('GET', '/oversight/stats')
}

// GET /oversight/difficulty_curve — pass@k time series
export async function getDifficultyCurve() {
  return request('GET', '/oversight/difficulty_curve')
}
