# agentflow - Improvements Roadmap

## 1. Root Directory Reorganization

Root dizinde 37 Go dosyasi var. Core agent loop ve temel interface'ler disindaki extension modulleri alt paketlere tasinmali.

### Mevcut Durum (root: 37 dosya)
```
agentflow/
  agent.go, config.go, tool.go, message.go, event.go, hook.go,
  permission.go, provider.go, session.go, budget.go, result.go,
  compactor.go, errors.go, doc.go, streaming_executor.go, subagent.go,
  compactor_sliding.go, compactor_staged.go, compactor_summary.go,
  team.go, skill.go, task.go, trigger.go, observability.go, plan_mode.go
  + 12 test dosyasi
```

### Hedef Yapi
```
agentflow/                    # Core: 16 kaynak + test dosyalari
  agent.go                    # Agent struct, Run(), RunSync(), RunSession(), Resume()
  config.go                   # Config, Option functions
  tool.go                     # Tool interface, ExecutionMode, Locality
  message.go                  # Message, ContentBlock, Role
  event.go                    # Event discriminated union
  hook.go                     # Hook interface, HookFunc
  permission.go               # PermissionChecker, AllowAll, DenyList, etc.
  provider.go                 # Provider, Stream, StreamEvent
  session.go                  # SessionStore interface, Session struct
  budget.go                   # TokenBudget, budgetTracker
  result.go                   # ResultLimiter, TruncateLimiter, HeadTailLimiter
  compactor.go                # Compactor interface (interface tanimini root'ta tut)
  errors.go                   # Sentinel errors, ProviderError, ToolError
  doc.go                      # Package documentation
  streaming_executor.go       # streamingToolExecutor (Agent internals kullanir)
  subagent.go                 # SubAgent (Agent method'lari)

  compactor/                  # Compactor implementasyonlari
    sliding.go                # SlidingWindowCompactor, TokenWindowCompactor
    summary.go                # SummaryCompactor
    staged.go                 # StagedCompactor

  team/                       # Team/Swarm koordinasyonu
    team.go                   # Team, TeamMember, Mailbox, SharedMemory, tools

  observability/              # Tracing ve cost tracking
    tracer.go                 # Trace, Span, Tracer
    cost.go                   # CostTracker, ModelPricing

  trigger/                    # Zamanlanmis calistirma
    trigger.go                # Trigger, TriggerScheduler

  plan/                       # Plan modu
    plan.go                   # Plan(), PlanAndExecute(), ExtractMemories()

  skill/                      # Skill sistemi
    skill.go                  # Skill, SkillRegistry, ExecuteSkill, ParseSkill

  task/                       # Task yonetimi
    task.go                   # Task, TaskStore, TaskStatus

  provider/                   # (degisiklik yok)
  internal/sse/               # (degisiklik yok)
  session/                    # (degisiklik yok)
  middleware/                  # (degisiklik yok)
  tools/                      # (degisiklik yok)
  _examples/                  # (degisiklik yok)
```

**Not:** `subagent.go` ve `streaming_executor.go` root'ta kalmali -- Agent struct'in private field'larina erisiyorlar.

---

## 2. Input Validation (JSON Schema)

Tool input'lari icin framework seviyesinde JSON Schema validation eksik. Suan model'den gelen input dogrudan tool'a iletiliyor -- hatali input'lar tool Execute() icinde yakalanmak zorunda.

**Yapilacak:**
- `internal/jsonschema/` paketi: temel JSON Schema validasyonu (type, required, enum, min/max)
- `executeSingleTool()` icinde tool.InputSchema()'ya karsi validation
- Validation hatasi durumunda modele anlamli hata mesaji dondurme
- Mevcut tool'lar etkilenmemeli (validation opsiyonel veya opt-out)

---

## 3. Provider-Level Rate Limiting

Provider seviyesinde rate limiting mekanizmasi yok. Ozellikle integration testlerde ve production'da API rate limit'lerine takilma riski var.

**Yapilacak:**
- `RateLimiter` interface tanimla (core'da)
- Token bucket veya sliding window implementasyonu
- `WithRateLimiter(limiter)` Option fonksiyonu
- `createStreamWithRetry()` icine rate limiter entegrasyonu
- Provider bazli rate limit konfigurasyonu

---

## 4. Structured Logging

Core agent loop'ta structured logging destegieksik. Middleware'deki `logging.go` sadece tool execution'lari logluyor; model cagrilari, retry'lar, compaction ve budget olaylari icin log yok.

**Yapilacak:**
- `WithLogger(logger *slog.Logger)` Option fonksiyonu
- Agent loop icinde kritik noktalarda structured log:
  - Model cagrisi baslangic/bitis (duration, token count)
  - Retry denemesi (attempt number, error, delay)
  - Compaction (before/after message count)
  - Budget uyarilari
  - Tool execution hatalari
- Default: no-op (sifir overhead)

---

## 5. Context Propagation (Trace Context)

Provider'lara kadar trace context tasima eksik. Observability modulu span'lar uretiyor ama bunlar provider HTTP cagrilarina propagate edilmiyor.

**Yapilacak:**
- Request struct'a `Metadata map[string]string` field ekle
- Tracer hook'larindan trace ID/span ID'yi context'e koy
- Provider implementasyonlarinda HTTP header'lara trace context ekle (`traceparent`, `tracestate`)
- Bu, production ortaminda end-to-end tracing saglar

---

## 6. Middleware Iyilestirmeleri

### 6a. Timeout Middleware
Tool execution icin konfigure edilebilir timeout middleware'i yok. Suanki timeout sadece bash tool icinde hardcoded.

**Yapilacak:**
- `middleware.Timeout(duration)` hook: herhangi bir tool execution'i timeout'a bagla
- Tool bazli timeout konfigurasyonu

### 6b. Circuit Breaker Middleware
Surekli hata veren tool'lar icin circuit breaker yok.

**Yapilacak:**
- `middleware.CircuitBreaker(threshold, resetDuration)` hook
- Belirli sayida hata sonrasi tool'u gecici devre disi birak
- Half-open state ile recovery

### 6c. Retry Middleware (Tool-Level)
Provider-level retry var ama tool-level retry yok.

**Yapilacak:**
- `middleware.Retry(maxRetries, backoff)` hook
- Belirli hata tipleri icin tool execution'i tekrarla

---

## 7. Agent Lifecycle Events

Agent'in lifecycle'inda eksik event'ler var:
- `EventCompaction` - compaction gerceklestiginde
- `EventRetry` - provider retry denemesinde
- `EventPermissionDenied` - permission reddedildiginde
- `EventHookBlocked` - hook tarafindan bloklama

**Yapilacak:**
- Yeni EventType sabitleri
- Ilgili event struct'lari
- agent.go icinde emit noktalarini ekle

---

## 8. Tool Input/Output Type Safety (Generics)

`InputSchema()` methodu `map[string]any` donduruyor -- type-safe degil. Go 1.23+ ile generic tool wrapper yazilabilir.

**Yapilacak:**
- `TypedTool[I, O any]` generic wrapper
- Input struct'indan otomatik JSON Schema uretimi
- `json.RawMessage` -> typed struct donusumu otomatik
- Mevcut `Tool` interface ile geriye uyumlu

---

## 9. Hooks Icin Multi-Phase Destek

Bir hook sadece tek bir phase'e baglanabiliyor (`Phase() HookPhase`). Ayni hook'un birden fazla phase'de calismasini istiyorsaniz duplicate etmek gerekiyor.

**Yapilacak:**
- `MultiPhaseHook` interface: `Phases() []HookPhase`
- `hooksForPhase()` icinde MultiPhaseHook kontrolu
- Mevcut `Hook` interface geriye uyumlu kalir

---

## 10. Error Recovery Stratejileri

Suan tool hata verdiginde sonuc modele "error" olarak iletiliyor. Ama framework seviyesinde konfigure edilebilir recovery stratejisi yok.

**Yapilacak:**
- `ErrorStrategy` interface: `OnToolError(ctx, call, err) (*ToolResult, Action)`
- Stratejiler: RetryN, Fallback(altTool), SkipAndContinue, AbortTurn
- `WithErrorStrategy(strategy)` Option

---

## 11. Agent Cloning / Forking

Mevcut agent konfigurasyonundan yeni agent turetme destegi sinirli. `buildChild()` private.

**Yapilacak:**
- `Agent.Clone(opts ...Option) *Agent` - mevcut agent'i kopyala ve uzerine yaz
- Farkli prompt/tool setleriyle ayni base konfigurasyondan agent'lar turet
- SubAgentConfig yerine daha esnek bir pattern

---

## 12. Provider Health Check

Provider'larin erisilebilir olup olmadigini kontrol eden mekanizma yok. Fallback provider reaktif (hata sonrasi gecis); proaktif health check yok.

**Yapilacak:**
- `HealthChecker` opsiyonel interface: `HealthCheck(ctx) error`
- `fallback.Provider`'a periyodik health check entegrasyonu
- Sagliksiz provider'lari gecici olarak devre disi birakma

---

## 13. Event Filtering / Subscription

Tum event'ler tek kanal uzerinden akiyor. Consumer sadece belirli event tiplerini dinlemek istediginde bile tum event'leri islemek zorunda.

**Yapilacak:**
- `EventFilter` type: `func(Event) bool`
- `Agent.RunFiltered(ctx, messages, filter)` veya wrapper utility
- `FilterEvents(ch <-chan Event, types ...EventType) <-chan Event` helper

---

## 14. Compactor Icin Metrics

Compaction ne zaman ve neden tetiklendi, kac mesaj silindi/ozetlendi bilgisi yok.

**Yapilacak:**
- `CompactionResult` struct: Before/After message count, strategy used, duration
- `Compactor.Compact()` return type'a opsiyonel metrics ekle (veya event olarak emit)
- Observability entegrasyonu

---

## 15. Test Coverage Iyilestirmeleri

### 15a. Provider Unit Testleri
OpenAI, Anthropic, Gemini provider'larinin unit testleri eksik/yetersiz. Integration test'ler API key gerektiriyor.

**Yapilacak:**
- Her provider icin mock HTTP server ile unit test
- SSE stream parsing edge case'leri
- Hata senaryolari (timeout, malformed response, partial stream)

### 15b. Race Condition Testleri
Concurrent tool execution ve event delivery icin race condition testleri sinirli.

**Yapilacak:**
- `-race` flag ile kapsamli test suite
- Concurrent SpawnChildren senaryolari
- Event channel backpressure testleri

### 15c. Benchmark Testleri
Performance benchmark'lari yok.

**Yapilacak:**
- `BenchmarkToolExecution` - tool pipeline overhead
- `BenchmarkEventDelivery` - event channel throughput
- `BenchmarkCompaction` - compaction strategy performansi
- `BenchmarkPartitionToolCalls` - batching algoritmasi

---

## 16. Documentation Iyilestirmeleri

### 16a. GoDoc Ornekleri
Exported fonksiyonlarin cogunlugunda `Example` testleri yok.

**Yapilacak:**
- `example_test.go` dosyasi: NewAgent, Run, RunSync, WithTool ornekleri
- Her provider icin usage ornekleri
- Middleware kullanim ornekleri

### 16b. Architecture Decision Records (ADRs)
Mimari kararlarin (neden pull-based stream, neden zero deps, neden functional options) dokumante edilmesi.

---

## Oncelik Sirasi

| # | Iyilestirme | Oncelik | Etki |
|---|------------|---------|------|
| 1 | Root Directory Reorganization | Yuksek | Maintainability |
| 2 | Input Validation | Yuksek | Guvenlik, Kararllik |
| 3 | Rate Limiting | Yuksek | Production readiness |
| 4 | Structured Logging | Orta | Observability |
| 5 | Context Propagation | Orta | Observability |
| 6 | Middleware Iyilestirmeleri | Orta | Kararllik |
| 7 | Lifecycle Events | Orta | Observability |
| 8 | Type Safety (Generics) | Dusuk | DX |
| 9 | Multi-Phase Hooks | Dusuk | DX |
| 10 | Error Recovery | Orta | Kararllik |
| 11 | Agent Cloning | Dusuk | DX |
| 12 | Provider Health Check | Orta | Production readiness |
| 13 | Event Filtering | Dusuk | DX |
| 14 | Compactor Metrics | Dusuk | Observability |
| 15 | Test Coverage | Yuksek | Kalite |
| 16 | Documentation | Orta | DX |
