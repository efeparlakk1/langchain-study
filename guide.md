# LangChain El Yazısı Rehberi

LangChain v1.0 — v1.3.15 (`>=1.0,<2`). Amaç: konseptleri ezberlemek değil, **elle yazabilir** hale gelmek.

Bu dosya müfredattır. Notebook’ları sen yazarsın. Kopyala-yapıştır çözüm yok; imza, zihinsel model ve görev var. Fark çıkan her yerde **v1.0** ve **v1.3.15** kullanımları yan yana durur.

---

## Bu rehberi nasıl kullanacaksın

1. Her ders için proje kökünde **tek bir notebook** aç: `00_setup.ipynb`, `01_messages.ipynb`, … Klasör ağacı yok.
2. Önce **zihinsel modeli** oku, sonra notebook görevlerini ekrana bakmadan yazmayı dene. Takılırsan bu dosyadaki imzalara dön.
3. Ders sonundaki **hafıza checkpoint**’ini geçmeden sonrakine geçme.
4. Fark bloğu varsa: ortak yolu yaz, sonra kendi sürümündeki varyantı çalıştır. Diğer sürümü de oku — “1.0’da atla” yok.

Resmi doküman (v1 hattı): [LangChain overview](https://docs.langchain.com/oss/python/langchain/overview)

---

## Sürüm sözleşmesi

`1.3.15`, 1.x içinde **additive** bir yama. `Runnable`, mesajlar, `init_chat_model`, `bind_tools`, `create_agent(model, tools, system_prompt=...)` 1.0’dan beri aynı.

Kurallar:

1. **Önce ortak yol** — her iki sürümde çalışan kod asıl görevdir.
2. **Fark varsa ikisi de yazılır** — ne değişti, neden eklendi, hangisini varsayılan alırsın.
3. Ara sürüm etiketi: bir özellik 1.1 veya 1.2’de geldiyse `1.1+` / `1.2+` yazılır; sen 1.0 veya 1.3.15 kullanıyorsun. Anlamı: **1.0’da yok, 1.3.15’te var.**
4. Checkpoint her zaman **ortak yola** bağlıdır.

Fark bloğu formatı (derslerde tekrarlanır):

```text
Ortak yol (1.0 ve 1.3.15):  ...
v1.0:                       ...
v1.3.15:                    ...   (gerekirse 1.1+ / 1.2+)
Fark:                       bir cümle
```

---

## Dual-safe import haritası

v1 namespace sadeleştirildi. `langchain.prompts` **yok**. Prompt ve runnable için `langchain_core` kullan.

| Ne | Import (1.0 ve 1.3.15) |
| --- | --- |
| Model başlatma | `from langchain.chat_models import init_chat_model` |
| Mesajlar | `from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage` |
| Prompt | `from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder` |
| Runnable | `from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda` |
| Parser | `from langchain_core.output_parsers import StrOutputParser` |
| Tool | `from langchain.tools import tool` |
| Agent | `from langchain.agents import create_agent` |
| Embeddings | `from langchain.embeddings import init_embeddings` |
| Splitter | `from langchain_text_splitters import RecursiveCharacterTextSplitter` |
| Document | `from langchain_core.documents import Document` |

`langchain.messages` ve `langchain.tools` 1.x boyunca `langchain_core` re-export’udur; ikisi de geçerli. Prompt/runnable’da `langchain_core` tercih et.

---

## Kullanma listesi (eski dünya)

Bunları yazma, tutorial’da görsen bile:

- `LLMChain`, `SequentialChain`, `ConversationChain`
- `AgentExecutor`, `initialize_agent`, `create_react_agent` (`langgraph.prebuilt`)
- `from langchain.prompts import ...` (v1’de yok)
- `langchain-classic` zincirleri, hub, eski retriever’lar
- Memory sınıfları (`ConversationBufferMemory` vb.) — v1’de yerini mesaj listesi + checkpointer alır

Yerine: LCEL (`|`), `bind_tools` + elle loop, sonra `create_agent`.

---

## Zihinsel harita (tüm kurs)

```text
mesaj listesi
    -> chat model (invoke / stream)
        -> prompt template (değişkenleri mesaja çevirir)
            -> structured output (şemalı cevap)
                -> Runnable / LCEL (her parça aynı sözleşme)
                    -> tool + elle loop (model araç seçer, sen çalıştırırsın)
                        -> create_agent (aynı loop’un harness’i)
                            -> RAG (retriever bir Runnable’dır)
                                -> memory (thread_id + checkpointer)
                                    -> LangGraph (ajan zaten bir graphtır)
```

LangChain = primitive’ler ve doğrusal zincirler.
LangGraph = state, döngü, dallanma. `create_agent` altında LangGraph vardır.
LangSmith = iz / debug (opsiyonel; kursun parçası değil).

---

# Ders 00 — Kurulum ve zihinsel harita

**Notebook:** `00_setup.ipynb`

### Amaç

Çalışan bir 1.x ortamı, sürümünü bilmek, `init_chat_model` ile tek satırda model açmak.

### Zihinsel model

LangChain model sınıfı koleksiyonu değil: **mesaj alıp mesaj üreten** standart bir arayüz. Provider paketi (`langchain-openai` vb.) o arayüzü doldurur. `init_chat_model` string’den doğru sınıfı seçer.

### Import

```python
from langchain.chat_models import init_chat_model
```

### Notebook görevleri

1. venv aç, Python 3.10+.
2. Kur:

```text
pip install "langchain>=1.0,<2" langchain-openai langchain-text-splitters
# mevcut yama: pip install langchain==1.3.15
```

Provider’ın başka olsun: `langchain-anthropic` veya `langchain-google-genai`.

3. `.env` veya ortam değişkeni: `OPENAI_API_KEY` (veya provider’ının anahtarı). Notebook’ta `os.environ` veya `python-dotenv`. Anahtarı git’e koyma.
4. Sürüm yazdır:

```python
import langchain
print(langchain.__version__)
```

`1.0.x` veya `1.3.15` — ikisi de bu rehber için geçerli.

5. Modeli aç, tek `invoke` at, dönen nesnenin **tipini** yazdır (`AIMessage` beklenir, düz `str` değil).

### Ortak yol — `init_chat_model`

```python
model = init_chat_model("openai:gpt-4o-mini", temperature=0)
# eşdeğer: init_chat_model("gpt-4o-mini", model_provider="openai")
msg = model.invoke("ping")
```

Model id’yi kendi hesabına göre değiştir. Önemli olan `"provider:model"` biçimi.

### Sürüm bloğu — provider string’leri

**Ortak yol (1.0 ve 1.3.15):**

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-4o-mini")
# anthropic: init_chat_model("anthropic:claude-sonnet-4-6")  # id'yi hesabına göre seç
# google:    init_chat_model("google_genai:gemini-2.5-flash-lite")
```

**v1.0:**

`init_chat_model` OpenAI / Anthropic / Google / Azure / Bedrock / Ollama vb. entegrasyon paketleriyle çalışır. LangSmith bir **model provider** olarak bu listede yoktur. Tracing ayrıdır: `LANGSMITH_TRACING=true` + API key (opsiyonel).

**v1.3.15:**

Aynı string biçimi durur. **1.3.15 eki:** LangSmith, `init_chat_model` provider’ı olarak eklendi. Tracing hâlâ ayrı bir konu; bu satır *LangSmith üzerinden model çağırmak* içindir (paket/hesap gerektirir):

```python
# 1.3.15 — 1.0'da ImportError / bilinmeyen provider
model = init_chat_model("langsmith:...")  # model id LangSmith dokümanındaki güncel ada göre
```

**Fark:** Çekirdek kullanım `"openai:..."`. LangSmith provider yalnızca 1.3.15. 1.0’da yok; 1.3.15’te varsa dene, yoksa OpenAI string’ine dön. Checkpoint: `provider:model` + `invoke`.

### Hafıza checkpoint’i

Ekrana bakmadan:

- `init_chat_model` import satırı
- `"openai:gpt-4o-mini"` ile model açma
- `invoke` sonucu `AIMessage` mi, `str` mi?

### Tuzaklar

- `from langchain.llms import OpenAI` — completion API, v1 kursunun parçası değil. Chat model kullan.
- `langchain.prompts` import’u v1’de patlar.
- `langchain` 0.3 ile 1.x karışık kurulum: `pip show langchain langchain-core` bak, major 1 olsun.

### Doküman

- [Install](https://docs.langchain.com/oss/python/langchain/install)
- [init_chat_model](https://docs.langchain.com/oss/python/langchain/models)

---

# Ders 01 — Messages

**Notebook:** `01_messages.ipynb`

### Amaç

LLM’e giden her şeyin bir **mesaj listesi** olduğunu elle kurmak.

### Zihinsel model

Tek bir string prompt yok. Konuşma sıralı rollerdir:

| Sınıf | Kim | Ne işe yarar |
| --- | --- | --- |
| `SystemMessage` | sen (geliştirici) | davranış, kurallar |
| `HumanMessage` | kullanıcı | girdi |
| `AIMessage` | model | metin, `tool_calls`, metadata |
| `ToolMessage` | sen (runtime) | bir tool çağrısının sonucu; `tool_call_id` şart |

Model `list[BaseMessage]` alır, bir `AIMessage` döner. Tool loop’ta bu listeye `AIMessage` + `ToolMessage` eklenir, tekrar `invoke` edilir.

`content` çoğu zaman `str`; multimodal / reasoning için liste veya `content_blocks` (standart bloklar, v1.0’dan beri).

### Import

```python
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
```

(`langchain_core.messages` aynı nesneler.)

### Notebook görevleri

1. Üç mesajlık bir liste kur: system, human, (elle yazılmış sahte) AI. Alanları tek tek yazdır: `.type`, `.content`.
2. Aynı listeyi `model.invoke(messages)` ile gönder. Dönen `AIMessage`’ın `content` ve varsa `usage_metadata` değerine bak.
3. `content_blocks` üzerinde dön: her bloğun `"type"` anahtarı (`text`, varsa `reasoning`, `tool_call`).
4. Bir `ToolMessage` örneği **kur** (henüz gerçek tool yok): `content="72F"`, `tool_call_id="call_1"`. Listenin neresine konacağını yorum satırında açıkla (AIMessage’daki tool call’dan **sonra**).

### Sürüm bloğu

**1.0 ve 1.3.15 aynı.** Dört mesaj sınıfı, `content_blocks`, `tool_calls` 1.0’dan beri çekirdek.

`content_blocks` provider kapsaması zamanla genişledi; yoksa boş/eksik olabilir — o zaman `.content` kullan. Sözleşme değişmedi.

### Hafıza checkpoint’i

Dört sınıfı import et, üç mesajlık listeyi kur, `invoke` et. `ToolMessage`’ın neden `tool_call_id` istediğini bir cümleyle yaz.

### Tuzaklar

- Düz string `invoke("hi")` çalışır (içeride `HumanMessage`’a sarılır) ama tool loop’ta **liste** şart.
- `AIMessage(content=...)` uydurmak ile modelden gelen `AIMessage` aynı tip; `tool_calls` yalnızca gerçek model çıktısında dolu olur.
- Role dict `{"role": "user", "content": "..."}` de kabul edilir. Checkpoint’te **sınıf** kullan; dict’i sonra görürsün (`create_agent` invoke’unda).

### Doküman

- [Messages](https://docs.langchain.com/oss/python/langchain/messages)

---

# Ders 02 — Chat models

**Notebook:** `02_chat_models.ipynb`

### Amaç

Aynı model nesnesinin dört çağrı biçimini ve `AIMessage` alanlarını elle kullanmak.

### Zihinsel model

Chat model bir `Runnable`’dır. Sözleşme:

| Metot | Ne |
| --- | --- |
| `invoke` | bir girdi → bir çıktı |
| `batch` | N girdi → N çıktı |
| `stream` | token / chunk üreteci |
| `ainvoke` / `astream` | async karşılıkları |

Girdi: `str` veya `list[BaseMessage]`. Çıktı: `AIMessage` (stream’de `AIMessageChunk`).

Okunacak alanlar: `content`, `tool_calls`, `usage_metadata`, `response_metadata`, `content_blocks`.

### Import

```python
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
```

### Notebook görevleri

1. `invoke` ile kısa bir soru. Tip + `content` + `usage_metadata`.
2. `stream` ile aynı soru; chunk’ları birleştir. Stream’de tam `usage_metadata` genelde **sonda** gelir.
3. `batch` ile 3 kısa prompt.
4. `await model.ainvoke(...)` (notebook’da `await` yeter).
5. Aynı soruyu `str` ve `[HumanMessage(...)]` ile at; çıktı tipinin aynı kaldığını doğrula.

### Sürüm bloğu — `model.profile`

**Ortak yol (1.0 ve 1.3.15):**

```python
model = init_chat_model("openai:gpt-4o-mini", temperature=0)
print(model.invoke("Merhaba").content)
for chunk in model.stream("Say count to 5"):
    print(chunk.content, end="")
```

**v1.0:**

Yetenek keşfi yok / resmi `profile` yok. Ne desteklendiğini provider dokümanından bilirsin (`bind_tools`, structured output, vizyon).

**v1.3.15 (1.1+):**

```python
# 1.1+ — 1.0'da AttributeError olabilir
print(model.profile)
# models.dev kaynaklı: context window, tool calling, structured output vb.
```

**Fark:** Çağrı yüzeyi aynı. `profile` 1.1+ keşif verisi; 1.0’da yok. Checkpoint: dört metot, `AIMessage` alanları. `profile` bonus.

### Hafıza checkpoint’i

`invoke` / `stream` / `batch` / `ainvoke` imzalarını ezberden yaz. Stream çıktısının neden `AIMessage` değil `AIMessageChunk` olduğunu söyle.

1.0’da `profile` yok; 1.3.15’te `model.profile`.

### Tuzaklar

- Stream’de `chunk.content` bazen `""` — birleştirerek bak.
- `temperature=0` derslerde tekrarlanabilirlik için.
- Modeli `bind_tools` ile sarmaladıktan sonra orijinal `model` değişmez; yeni nesne döner.

### Doküman

- [Models](https://docs.langchain.com/oss/python/langchain/models)

---

# Ders 03 — Prompt templates

**Notebook:** `03_prompts.ipynb`

### Amaç

Değişkenleri mesaj listesine çeviren şablonu LCEL’e takmak.

### Zihinsel model

`ChatPromptTemplate` bir `Runnable`’dır. Girdi: `dict` (`{"topic": "çay"}`). Çıktı: `ChatPromptValue` / mesaj listesi. Model hâlâ mesaj ister; şablon **köprüdür**.

`MessagesPlaceholder("history")` listenin ortasına değişken uzunlukta mesaj sokar — memory’nin tohumu.

### Import

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
```

### Notebook görevleri

1. `from_messages` ile system + human şablonu. `{topic}` değişkeni. `.invoke({"topic": "..."})` çıktısına bak (mesajlar).
2. Aynı şablonu `prompt | model` diye bağla, `invoke` et. Parser henüz yok; çıktı `AIMessage`.
3. `MessagesPlaceholder("history")` ekle. `history`’ye önceki `HumanMessage`/`AIMessage` listesi ver. Placeholder’sız invoke’un neden hata verdiğini gör (veya `optional` ise boş geç).
4. Few-shot: şablona sabit `("human", ...)`, `("ai", ...)` çiftleri koy. Değişken few-shot’a henüz girme.

### Sürüm bloğu

**1.0 ve 1.3.15 aynı.** `ChatPromptTemplate.from_messages`, `MessagesPlaceholder` `langchain_core`’da durur.

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen {role} olarak cevap verirsin."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])
```

### Hafıza checkpoint’i

`from_messages` ile 2 rollerli şablon + `prompt | model` + `dict` invoke. `from langchain.prompts import ChatPromptTemplate` yazarsan bu dersi geçme — v1’de yok.

### Tuzaklar

- `PromptTemplate` (düz string) chat modellerle yetmez; `ChatPromptTemplate` kullan.
- `invoke("metin")` şablona gitmez; anahtarları olan `dict` gerekir.
- `f-string` vs şablon: LangChain `{var}` şablon değişkenidir. Metinde gerçek süslü parantez varsa kaçış kurallarına dikkat.

### Doküman

- [ChatPromptTemplate](https://reference.langchain.com/python/langchain-core/prompts/ChatPromptTemplate)

---

# Ders 04 — Structured output

**Notebook:** `04_structured_output.ipynb`

### Amaç

Serbest metin yerine **doğrulanmış nesne** almak. Agent stratejilerini ayrı, model yöntemini ayrı görmek.

### Zihinsel model

Model varsayılanı düz metindir. Downstream kod `str` parse etmek istemez. İki katman:

1. **Model katmanı (ortak yol):** `with_structured_output(Schema)` — yeni bir Runnable; `invoke` doğrudan Pydantic örneği (veya dict) döner.
2. **Agent katmanı:** ajan döngüsü bittiğinde şemalı cevap. v1’de `response_format` + `ToolStrategy` / `ProviderStrategy`.

Bu derste (1) zorunlu. (2) okunur ve 1.0 vs 1.3.15 diye yazılır; ajanı ders 09’da kurarsın.

### Import

```python
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
```

Agent farkı için (şimdilik sadece oku / sonra dene):

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
```

`ProviderStrategy` import’u 1.0’da yoksa: **1.0’da yok, 1.3.15’te var.** `ToolStrategy` 1.0 agent API’sinde vardır.

### Notebook görevleri

1. Küçük bir Pydantic model (`title: str`, `year: int`, `tags: list[str]`). Field description yaz — şema modele gider.
2. `structured = model.with_structured_output(Film)` sonra `invoke`. Tipin `Film` olduğunu assert et.
3. Bilerek bozuk bir senaryo dene (imkânsız şema / saçma girdi). Hatanın nerede patladığını not et.
4. Aşağıdaki sürüm bloğunu oku. 1.3.15’teysen `create_agent` + `ToolStrategy` **ve** `ProviderStrategy`’yi ayrı hücrelerde dene. 1.0’daysan `ToolStrategy` dene; `ProviderStrategy` hücresini yorum + “1.0’da yok” notuyla bırak.

### Sürüm bloğu — model vs agent şema

**Ortak yol (1.0 ve 1.3.15) — chat model:**

```python
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

class Film(BaseModel):
    title: str = Field(description="Film adı")
    year: int = Field(description="Vizyon yılı")

model = init_chat_model("openai:gpt-4o-mini", temperature=0)
structured = model.with_structured_output(Film)
film = structured.invoke("Inception hakkında kısa yapılandırılmış özet")
# film: Film
```

**v1.0 — agent `response_format`:**

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],  # veya gerçek tool listesi
    response_format=ToolStrategy(Film),
)
result = agent.invoke({"messages": [{"role": "user", "content": "Inception"}]})
# result["structured_response"] -> Film
```

`ToolStrategy`: model şemayı **tool call** gibi üretir. 1.0’da agent structured output’un temel yolu budur. `with_structured_output` agent’sız zincirler içindir; ikisini karıştırma.

**v1.3.15 (1.1+, sıkı şema 1.2+):**

```python
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy

# Native provider structured output (json_schema vb.)
agent_native = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    response_format=ProviderStrategy(Film),
)

# ToolStrategy hâlâ geçerli — fallback / provider native yoksa
agent_tool = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    response_format=ToolStrategy(Film),
)
```

1.1+ bazı modellerde `ProviderStrategy` `model.profile` ile çıkarılabilir. 1.2+ sıkı schema adherence (`strict`) bu hat üzerindedir.

**Fark:** Zincirde her zaman `with_structured_output`. Agent’ta 1.0 → `ToolStrategy`; 1.3.15 → `ToolStrategy` **veya** `ProviderStrategy`. Varsayılan öğrenme: önce model katmanı. Native strateji provider’a bağlıdır.

### Hafıza checkpoint’i

Pydantic + `with_structured_output` + `invoke` → model örneği.

1.0 agent: `response_format=ToolStrategy(Schema)`.
1.3.15 agent: `ToolStrategy` veya `ProviderStrategy`.

### Tuzaklar

- `StrOutputParser` structured output değildir; sadece `.content` string’i.
- `include_raw=True` hem ham `AIMessage` hem parse edilmiş nesneyi verir — debug için.
- Schema field’larında description yoksa model daha çok uydurur.

### Doküman

- [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output)

---

# Ders 05 — Runnable ve LCEL

**Notebook:** `05_lcel.ipynb`

### Amaç

LangChain’in çekirdek soyutlamasını kas hafızasına almak: her parça `Runnable`, birleştirme `|`.

### Zihinsel model

`prompt`, `model`, `parser`, `retriever`, lambda — hepsi aynı protokol:

```text
.invoke(input) -> output
.batch([...])
.stream(input)
.ainvoke(input)
```

`chain = prompt | model | parser` bir `RunnableSequence` üretir. Ortadaki çıktı sağdakinin girdisidir.

Tip akışı (ezberle):

```text
dict  -> ChatPromptTemplate -> mesajlar
      -> ChatModel          -> AIMessage
      -> StrOutputParser    -> str
```

Bu yüzden parser’ı sona koymazsan zincir `AIMessage` bırakır.

### Import

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
```

### Notebook görevleri

1. Üç parçayı ayrı `invoke` et: prompt dict alsın, model mesaj alsın, parser `AIMessage` alsın. Her adımın girdi/çıktı tipini yaz.
2. Aynı üçünü `|` ile birleştir, tek `invoke({"topic": "..."})`.
3. Zincir üzerinde `.get_graph().print_ascii()` (veya `.get_graph()`) — sırayı gör.
4. Parser’sız `prompt | model` ile parser’lı karşılaştırmanın tiplerini not et.

### Sürüm bloğu

**1.0 ve 1.3.15 aynı.** LCEL 1.x’in omurgası. Yeni sürüm burada sözleşme değiştirmez.

```python
chain = prompt | model | StrOutputParser()
text = chain.invoke({"topic": "LCEL"})
```

### Hafıza checkpoint’i

Boş notebook: şablon, model, parser, `|`, `invoke`. Ara tipleri (dict / messages / AIMessage / str) ezberden söyle.

### Tuzaklar

- `|` soldan sağa. `model | prompt` anlamsız.
- Eski `LLMChain(prompt=, llm=)` yazma.
- `StrOutputParser` structured output değildir.

### Doküman

- [Component architecture](https://docs.langchain.com/oss/python/langchain/component-architecture)

---

# Ders 06 — Composition

**Notebook:** `06_composition.ipynb`

### Amaç

Tek hat dışında paralel, passthrough, lambda, assign, fallback yazmak.

### Zihinsel model

| Parça | Ne yapar |
| --- | --- |
| `RunnablePassthrough()` | girdiyi olduğu gibi iletir |
| `RunnableLambda(fn)` | düz Python’u Runnable yapar |
| `RunnableParallel({...})` veya `{...}` dict’i `|` solunda | aynı girdiyi birkaç kola |
| `.assign(key=runnable)` | mevcut dict’e anahtar ekler |
| `.with_fallbacks([other])` | hata / boşlukta yedek |

Klasik RAG iskeleti bu derstedir (retriever henüz sahte lambda olabilir):

```text
{"context": retriever_or_lambda, "question": RunnablePassthrough()}
| prompt | model | parser
```

Dict literal LCEL’de `RunnableParallel`’e döner.

### Import

```python
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
```

### Notebook görevleri

1. `RunnableLambda` ile `lambda x: x["q"].upper()` — `invoke({"q": "hi"})`.
2. `RunnableParallel` veya dict: `{"len": lambda..., "echo": RunnablePassthrough()}`.
3. `.assign(summary= prompt | model | parser)` — orijinal anahtarlar duruyor mu bak.
4. İki model / iki zincir: `primary.with_fallbacks([backup])`. Backup’ı kasten çalışan basit bir lambda yap; primary’yi bilinçli boz (yanlış model adı) ve fallback’in devreye girdiğini gör.
5. `RunnableLambda` ile `if` dallanması (kısa girdi → kısa zincir, uzun → uzun). Branch’in “graph” olmadığını; hâlâ düz Python + Runnable olduğunu not et. Gerçek döngü LangGraph’tadır.

### Sürüm bloğu

**1.0 ve 1.3.15 aynı.**

```python
chain = (
    {"question": RunnablePassthrough(), "n": RunnableLambda(lambda q: len(q))}
    | prompt
    | model
    | StrOutputParser()
)
```

### Hafıza checkpoint’i

Passthrough + Parallel dict + Lambda + `|` model. RAG iskeletini retriever olmadan (sabit string context) yaz.

### Tuzaklar

- Parallel kolları **aynı girdiyi** alır. Sol kol `retriever`, sağ kol `Passthrough` ise girdi her ikisine de gider — bu yüzden RAG’de `question` Passthrough ile korunur.
- Lambda içinde ağır iş / I/O: tracing ve `batch` için Runnable kal, çıplak yan etkiyi sınırla.
- Fallback sonsuz zincirleme: bir yedek yeter.

### Doküman

- [langchain-core runnables](https://reference.langchain.com/python/langchain_core/runnables/)

---

# Ders 07 — Tools ve elle tool loop

**Notebook:** `07_tools.ipynb`

### Amaç

Ajan sihrinden **önce** tool calling’i Python ile yazmak. `create_agent` bu derste yok.

### Zihinsel model

Tool = JSON şemalı fonksiyon. Model fonksiyonu **çalıştırmaz**; “şu isim, şu argümanlar” der (`AIMessage.tool_calls`). Sen çalıştırır, `ToolMessage` ile geri verir, tekrar `invoke` edersin. Bitene kadar (tool_calls boş) döngü.

```text
messages = [HumanMessage(...)]
while True:
    ai = model_with_tools.invoke(messages)
    messages.append(ai)
    if not ai.tool_calls:
        break
    for call in ai.tool_calls:
        result = tool_map[call["name"]].invoke(call["args"])
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
```

Bu, `create_agent`’ın gizlediği şeydir.

### Import

```python
from langchain.tools import tool
from langchain.messages import HumanMessage, ToolMessage
```

### Notebook görevleri

1. `@tool` ile `add(a: int, b: int) -> int`. Docstring yaz — model bunu okur. `add.name`, `add.description`, `add.args` (veya schema) yazdır.
2. `model_with_tools = model.bind_tools([add])`. “3 ile 5’i topla” de. `ai.tool_calls` yapısını incele: `name`, `args`, `id`.
3. Yukarıdaki `while` döngüsünü **kendi başına** yaz. `tool_map` dict. Final `ai.content` sayı/cevap içermeli.
4. İkinci bir tool (`multiply`) ekle. Modelin hangisini seçtiğini loop log’unda gör.
5. Sürüm bloğundaki `extras` örneğini oku. 1.3.15’teysen bir tool’a `extras` koyup schema/dump’a yansıyanı incele. 1.0’daysan AttributeError’u not et.

### Sürüm bloğu — tool `extras`

**Ortak yol (1.0 ve 1.3.15):**

```python
from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
    """İki tam sayıyı topla."""
    return a + b

model_with_tools = model.bind_tools([add])
ai = model_with_tools.invoke("2+3 nedir, tool kullan")
print(ai.tool_calls)
```

**v1.0:**

Tool şeması ad, açıklama, parametreler. Provider’a özel bayrak (`extras`) yok. Anthropic programmatic calling / client-side built-in tools bu sürümede `@tool` ile taşınmaz.

**v1.3.15 (1.2+):**

```python
@tool
def search(query: str) -> str:
    """Web'de ara (örnek)."""
    return "ok"

# 1.2+ — provider-specific; 1.0'da extras yok
search.extras = {"some_provider_flag": True}  # gerçek anahtarlar provider dokümanında
# örn. Anthropic tool search / programmatic calling, OpenAI built-in tools
```

**Fark:** Loop ve `bind_tools` aynı. `extras` 1.2+ ile provider’a özel tool meta. Checkpoint: `@tool` + `bind_tools` + elle `while` + `ToolMessage(tool_call_id=...)`. `extras` ikinci kullanım.

### Hafıza checkpoint’i

Decorator, bind, `tool_calls` okuma, `ToolMessage` ekleme, döngü. `create_agent` yazarsan bu dersi sıfırla.

1.0’da `extras` yok; 1.3.15’te (1.2+) var.

### Tuzaklar

- `ToolMessage`’ı `tool_call_id` olmadan eklemek bir sonraki `invoke`’u bozar.
- Docstring yoksa model tool’u yanlış / hiç seçmez.
- `tool.invoke` args dict bekler (`{"a": 1, "b": 2}`), positional değil.
- Sonsuz döngü: max 5 iterasyon koy.

### Doküman

- [Tools](https://docs.langchain.com/oss/python/langchain/tools)
- [Tool calling](https://docs.langchain.com/oss/python/langchain/models#tool-calling)

---

# Ders 08 — Agents (`create_agent`)

**Notebook:** `08_agents.ipynb`

### Amaç

Ders 07’deki loop’un harness’i olduğunu görmek; string prompt, stream ve middleware farklarını her iki sürümde okumak.

### Zihinsel model

```text
Agent = Model + Harness
Harness = system prompt + tools + middleware + (altta LangGraph)
```

`create_agent(...)` compiled graph döner. Invoke sözleşmesi:

```python
agent.invoke({"messages": [{"role": "user", "content": "..."}]})
# veya HumanMessage listesi
```

Çıktı state dict: `result["messages"]` — son eleman genelde final `AIMessage`.

Middleware: loop’un kancaları (`before_model`, `after_model`, `wrap_model_call`, `wrap_tool_call`). 1.0’da kavram + birkaç built-in; 1.1+ ek built-in’ler.

### Import

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import SystemMessage
```

### Notebook görevleri

1. Ders 07’deki `add` (ve isteğe bağlı `multiply`) ile:

```python
agent = create_agent(
    model="openai:gpt-4o-mini",  # veya init_chat_model(...) nesnesi
    tools=[add],
    system_prompt="Matematikte tool kullan, uydurma.",
)
```

`invoke` et, `result["messages"]` tiplerini sırayla yaz (Human, AI+tool_calls, Tool, AI).

2. Aynı soruyu ders 07 loop’unla ve `create_agent` ile sor. Mesaj izinin **aynı fikir** olduğunu gör (harness senin yazdığın while’dır).
3. `agent.stream(...)` — mode varsayılanı ne üretiyor bak (`updates` / `values` dokümana göre değişebilir). Token stream ayrıdır.
4. Sürüm bloklarındaki `SystemMessage` prompt, `astream_events` v3, middleware çiftlerini kendi sürümünde çalıştır / 1.0’da yoksa not düş.

### Sürüm bloğu — `system_prompt`

**Ortak yol (1.0 ve 1.3.15):**

```python
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[add],
    system_prompt="Sen yardımcı bir asistanısın. Hesap için tool kullan.",
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "17+25 nedir?"}]}
)
print(result["messages"][-1].content)
```

**v1.0:**

`system_prompt` **str** (veya `None`). `SystemMessage` vermek 1.0’da desteklenmez / hata.

**v1.3.15 (1.1+):**

```python
from langchain.messages import SystemMessage

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[add],
    system_prompt=SystemMessage(content="Sen yardımcı bir asistanısın."),
    # 1.1+: cache control, structured content blocks için SystemMessage gerekir
)
```

String hâlâ geçerlidir. `SystemMessage` ek güç (Anthropic cache vb.).

**Fark:** Varsayılan öğrenme string. 1.3.15’te aynı yerde `SystemMessage` da olur. Checkpoint: string.

### Sürüm bloğu — streaming

**Ortak yol (1.0 ve 1.3.15):**

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "17+25"}]},
):
    print(chunk)
```

Event stream (v1/v2, her iki hatta da var):

```python
async for ev in agent.astream_events(
    {"messages": [{"role": "user", "content": "17+25"}]},
    version="v2",
):
    print(ev["event"], ev.get("name"))
```

**v1.0:**

`stream` + `astream_events(version="v1"|"v2")`. `version="v3"` yok.

**v1.3.15 (1.3):**

```python
async for ev in agent.astream_events(
    {"messages": [{"role": "user", "content": "17+25"}]},
    version="v3",
):
    # content-block-centric; typed projeksiyonlar (messages, lifecycle, ...)
    print(ev)
```

**Fark:** Öğrenme `invoke` + `stream`. v2 events 1.0’da da var. v3 yalnızca 1.3. 1.0’da `version="v3"` geçersiz.

### Sürüm bloğu — middleware

**Ortak yol (1.0 ve 1.3.15) — kavram:**

Hook’lar: `before_agent`, `before_model`, `wrap_model_call`, `wrap_tool_call`, `after_model`, `after_agent`.

1.0 built-in örnekleri (isimler sürümde mevcutsa kullan; import hata verirse paketten kontrol et):

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[add],
    middleware=[
        SummarizationMiddleware(model="openai:gpt-4o-mini", trigger={"tokens": 500}),
    ],
)
```

`HumanInTheLoopMiddleware` checkpointer ister; bu derste sadece import + imzayı oku, takma zorunlu değil.

**v1.0:**

Yukarıdaki hook modeli + PII / summarization / HITL. Retry ve OpenAI moderation middleware yok.

**v1.3.15 (1.1+):**

```python
from langchain.agents.middleware import (
    SummarizationMiddleware,
    ModelRetryMiddleware,  # 1.1+ — 1.0'da yok
)

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[add],
    middleware=[
        ModelRetryMiddleware(),  # backoff; parametreleri dokümandan doldur
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            trigger={"tokens": 500},  # 1.1+: profile ile de tetiklenebilir
        ),
    ],
)
```

OpenAI content moderation middleware 1.1+ entegrasyon paketindedir; çekirdek `langchain.agents.middleware` dışında olabilir.

**Fark:** Loop kancaları 1.0’dan beri var. 1.3.15’te retry, profile-aware summarization, moderation ek. Checkpoint: `create_agent` + tools + string prompt + messages invoke. Middleware’i isimlendir, derin custom class yazma.

### Hafıza checkpoint’i

`create_agent(model=..., tools=..., system_prompt="...")` + `invoke({"messages": [...]})` + son mesajın content’i.

1.0: `system_prompt` str; events v2; retry middleware yok.
1.3.15: str veya `SystemMessage`; events v3 var; `ModelRetryMiddleware` var.

### Tuzaklar

- `agent.invoke("soru")` değil; state dict.
- `create_react_agent` / `AgentExecutor` yazma.
- Pre-bound `model.bind_tools`’u `create_agent(model=...)`’e vermek v1’de önerilmez; tools’u `create_agent(tools=)` ile ver.
- Final cevabı `result`’ın kendisi sanma; `messages[-1]`.

### Doküman

- [Agents](https://docs.langchain.com/oss/python/langchain/agents)
- [Middleware](https://docs.langchain.com/oss/python/langchain/middleware)
- [Event streaming](https://docs.langchain.com/oss/python/langchain/event-streaming)

---

# Ders 09 — RAG

**Notebook:** `09_rag.ipynb`

### Amaç

Dış metni retrieve edip LCEL ile modele bağlam olarak vermek. Ajan şart değil.

### Zihinsel model

```text
ham metin -> splitter -> Document listesi
         -> embeddings -> vector store
query    -> retriever.invoke(query) -> ilgili Document'lar
         -> prompt(context, question) -> model
```

Retriever bir Runnable’dır. Bu yüzden ders 06 iskeleti gerçeğe döner:

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
chain.invoke("soru")  # str in, str out
```

`format_docs`: `list[Document] -> str`.

### Import

```python
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import init_embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
```

Chroma istersen: `pip install langchain-chroma`, import’u paketin güncel yolundan al. İlk geçişte `InMemoryVectorStore` yeter — ekstra servis yok.

### Notebook görevleri

1. 3–5 `Document` uydur (kısa paragraflar, `metadata={"source": ...}`).
2. `RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)` — chunk sayısını yazdır.
3. `emb = init_embeddings("openai:text-embedding-3-small")` (provider’ına göre id). `InMemoryVectorStore.from_documents(chunks, emb)`.
4. `retriever = vs.as_retriever(k=2)`. `retriever.invoke("...")` Document döndüğünü gör.
5. `format_docs` lambda/fonksiyon. Prompt: “Sadece context’e dayan. Yoksa ‘bilmiyorum’.” Zinciri kur, **context’te olan** ve **olmayan** iki soru sor.

### Sürüm bloğu

**1.0 ve 1.3.15 aynı** (RAG primitive’leri `langchain_core` + `langchain_text_splitters` + embeddings).

```python
emb = init_embeddings("openai:text-embedding-3-small")
vs = InMemoryVectorStore.from_documents(chunks, emb)
retriever = vs.as_retriever()
```

`init_embeddings` v1 convenience API’sidir; 1.0 ve 1.3.15’te vardır. Eski `OpenAIEmbeddings()` da çalışır ama checkpoint `init_embeddings` + `init_chat_model` simetrisi.

### Hafıza checkpoint’i

Document → split → embed → store → retriever → `{"context": ..., "question": Passthrough}` → prompt → model. Dört-beş satırda iskeleti ezberden yaz.

### Tuzaklar

- Tüm corpus’u prompt’a gömmek RAG değildir.
- Chunk çok büyük: gürültü. Çok küçük: cümle kırılır. 200–800 karakterle başla.
- Retriever’ı agent tool yapmak ayrı desen; bu derste **zincir**.
- `from langchain.retrievers import ...` v1 çekirdeğinde yok (`langchain-classic`).

### Doküman

- [Component architecture — RAG](https://docs.langchain.com/oss/python/langchain/component-architecture)

---

# Ders 10 — Memory

**Notebook:** `10_memory.ipynb`

### Amaç

“Memory sınıfı” değil: **mesaj geçmişi + thread**.

### Zihinsel model

Kısa bellek = aynı `messages` listesini bir sonraki çağrıya vermek.

- Zincir: `MessagesPlaceholder("history")` + senin tuttuğun Python listesi.
- Agent / graph: checkpointer + `config={"configurable": {"thread_id": "..."}}`. Aynı `thread_id` aynı geçmiş.

Checkpointer olmadan her `invoke` amneziktir.

### Import

```python
from langgraph.checkpoint.memory import InMemorySaver  # paket adın langgraph
from langchain.agents import create_agent
```

Kurulumda `langgraph` zaten `langchain` 1.x bağımlılığıdır.

### Notebook görevleri

1. Saf Python: `history = []`. Her turda human+ai ekle, `MessagesPlaceholder`’lı zincire `{"history": history, "question": q}` ver. İkinci soruda birinciye atıf olsun (“onu kaç derece demiştim?”).
2. `create_agent(..., checkpointer=InMemorySaver())`.

```python
cfg = {"configurable": {"thread_id": "user-1"}}
agent.invoke({"messages": [{"role": "user", "content": "Adım Eren"}]}, config=cfg)
agent.invoke({"messages": [{"role": "user", "content": "Adım ne?"}]}, config=cfg)
```

3. Aynı agent, **farklı** `thread_id` — ad unutulmalı.
4. Checkpointer’sız aynı iki invoke — unutulmalı.

### Sürüm bloğu

**1.0 ve 1.3.15 aynı.** `thread_id` + checkpointer LangGraph sözleşmesi. `ConversationBufferMemory` kullanma.

### Hafıza checkpoint’i

İki cümle: (1) zincirde history listesi, (2) agent’ta checkpointer + `thread_id`. İkinci `invoke`’a önceki mesajları elle koymadan ismin hatırlandığını göster.

### Tuzaklar

- `thread_id` üretmeyi unutup her istekte yeni uuid = memory yok.
- `InMemorySaver` process ölünce silinir. Production’da Postgres/SQLite saver — bu kursun dışı.
- Tüm geçmişi sınırsız biriktirmek context’i patlatır; summarization middleware (ders 08) bunun için.

### Doküman

- [Short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory)

---

# Ders 11 — LangGraph minimum

**Notebook:** `11_langgraph.ipynb`

### Amaç

`create_agent` altındaki makineyi üç kavramla yazmak: state, node, edge. Multi-agent yok.

### Zihinsel model

```text
State  = turlar arasında duran TypedDict
Node   = state alan, kısmi update dönen fonksiyon
Edge   = sıradaki node (sabit veya conditional)
START -> ... -> END
compile() -> yine Runnable (invoke / stream)
```

Döngü = conditional edge’in eski bir node’a dönmesi. LCEL döngü yapamaz; graph yapar.

İlk graph’ın agent olması gerekmez: `{"text": str}` state, `upper` node, `END`. Sonra iki node + if.

### Import

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
```

### Notebook görevleri

1. `State(TypedDict)` tek alan `n: int`. Node `n+1` döndürsün. `add_node`, `add_edge(START, ...)`, `add_edge(..., END)`, `compile`, `invoke({"n": 0})`.
2. Conditional: `n < 3` ise aynı node’a (dikkat: sonsuz döngüye `n` artışı şart), değilse `END`. Veya iki node `even`/`odd`.
3. `MessagesState` veya `Annotated[list, add_messages]` ile tek model node’u: girdi mesaj, model `invoke`, `{"messages": [ai]}`. Bu, ajanın **toolsuz** hali.
4. `stream` et — her node update’ini gör.

### Sürüm bloğu — stream formatı

**Ortak yol (1.0 ve 1.3.15):**

```python
graph = builder.compile()
print(graph.invoke({"n": 1}))

for part in graph.stream({"n": 1}):
    print(part)  # varsayılan: dict, node adı -> update
```

**v1.0:**

Varsayılan dict stream / dict invoke. Typed `version="v2"` yok (LangGraph 1.1 öncesi).

**v1.3.15 (LangGraph 1.1+ — langchain 1.3.15 stack’inde gelir):**

```python
# opt-in; 1.0 langchain + eski langgraph'ta yok
result = graph.invoke({"n": 1}, version="v2")
# GraphOutput: .value, .interrupts — dict erişimi de bozulmadan durur

for part in graph.stream({"n": 1}, version="v2"):
    # StreamPart: type, ns, data
    print(part)
```

**Fark:** Öğrenme varsayılan dict. v2 typed format 1.3.15 hattında opt-in. Checkpoint: StateGraph + node + edge + compile + invoke. `version="v2"` ikinci kullanım.

### Hafıza checkpoint’i

Üç satırlık zihin: state tipi, `add_node`, `add_edge`/`add_conditional_edges`, `compile`, `invoke`. Neden `create_agent`’ın LCEL zinciri değil graph olduğu: **döngü**.

1.0: `stream` dict. 1.3.15: aynı + `version="v2"`.

### Tuzaklar

- Node tüm state’i kopyalamak zorunda değil; **kısmi update** (`{"n": 2}`).
- `add_messages` reducer: `messages=` üzerine yazmaz, append eder. Reducer’sız list ezilir.
- Aynı node’dan hem sabit edge hem conditional — davranış karmaşıklaşır; birini seç.
- Deep Agents / supervisor bu dersin dışı.

### Doküman

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [Use graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api)

---

# Kapanış — hafızadan yazma sınavı

Yeni bir notebook: `99_from_memory.ipynb`. Bu dosyayı kapat. Aşağıdakileri yaz. Takılınca ders numarasına dön, kopyalama.

1. `init_chat_model` + `"provider:model"` + `invoke` → `AIMessage`.
2. Dört mesaj sınıfından üç elemanlı liste.
3. `ChatPromptTemplate.from_messages` + `prompt | model | StrOutputParser`.
4. Pydantic + `with_structured_output`.
5. `RunnablePassthrough` + dict parallel ile sahte RAG iskeleti (context sabit string).
6. `@tool` + `bind_tools` + `tool_calls` + `ToolMessage` + `while` (max 5).
7. `create_agent(model, tools, system_prompt="...")` + `invoke({"messages": ...})`.
8. Document → split → `InMemoryVectorStore` → retriever → LCEL RAG.
9. `InMemorySaver` + `thread_id` ile iki tur.
10. `StateGraph` bir node, START→node→END, `compile`, `invoke`.

Sınav **ortak yol**. Bitince aşağıdaki farkları ezberden doldur (kod şart değil, birer satır):

| Konu | v1.0 | v1.3.15 |
| --- | --- | --- |
| `init_chat_model` LangSmith provider | yok | var (1.3.15) |
| `model.profile` | yok | var (1.1+) |
| Agent `system_prompt` | `str` | `str` veya `SystemMessage` (1.1+) |
| Agent structured | `ToolStrategy` | `ToolStrategy` veya `ProviderStrategy` (1.1+) |
| Tool `extras` | yok | var (1.2+) |
| `astream_events` | v1/v2 | v1/v2/v3 (1.3) |
| Middleware retry | yok | `ModelRetryMiddleware` (1.1+) |
| Graph `stream(..., version="v2")` | yok | opt-in (LangGraph 1.1+) |

---

## Artık bilmen gerekenler

- Konuşma = mesaj listesi
- Her parça Runnable; `|` sıra, dict parallel
- Tool = şema; çalıştıran sensin
- Agent = senin loop’unun graph harness’i
- RAG = retriever Runnable + prompt
- Memory = history veya `thread_id` + checkpointer
- Graph = state + node + edge; döngü burada

## Henüz gerekmeyenler

LangServe, Deep Agents, multi-agent supervisor, production checkpoint store, `langchain-classic`, onlarca vector DB entegrasyonu, custom middleware class hiyerarşisi.

1.3 eklerini (v3 events, `extras`, `ProviderStrategy`) **ortak yolun yanında** gördün; günlük varsayılanın hâlâ 1.0 ile aynı kesişim.

---

## Ders sırası (dosya adları)

```text
00_setup.ipynb
01_messages.ipynb
02_chat_models.ipynb
03_prompts.ipynb
04_structured_output.ipynb
05_lcel.ipynb
06_composition.ipynb
07_tools.ipynb
08_agents.ipynb
09_rag.ipynb
10_memory.ipynb
11_langgraph.ipynb
99_from_memory.ipynb
```

Hepsi proje kökünde. Başka klasör açma. Bir ders = bir notebook = o konsepti elle yazmak.
