"""Offline SDK/LangChain streams using saved consecutive-call usage samples."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import httpx
from langchain_core.messages import HumanMessage
import pytest

from app.core.llm import _CumulativeUsageChatOpenAI

# Same samples as the preserved usage-gap reproduction and eval-usage-v2 tests.
SAMPLES={'A':(382,31,128),'B':(874,50,640),'equal':(382,31,128)}


def body(label):
    incoming,outgoing,cache=SAMPLES[label]
    def chunk(usage=None,content=None):
        return {'id':'offline','object':'chat.completion.chunk','created':0,'model':'deepseek-flash',
            'choices':[] if content is None else [{'index':0,'delta':{'content':content},'finish_reason':None}],
            'usage':usage}
    usage={'prompt_tokens':incoming,'completion_tokens':outgoing,'total_tokens':incoming+outgoing,
        'prompt_cache_hit_tokens':cache,'prompt_tokens_details':{'cached_tokens':cache},
        'completion_tokens_details':{'reasoning_tokens':outgoing-1}}
    values=[chunk(content='ok'),chunk(usage),chunk(usage)]  # Duplicate cumulative snapshot.
    return b''.join(b'data: '+json.dumps(v).encode()+b'\n\n' for v in values)+b'data: [DONE]\n\n'


def label(request):return json.loads(request.content)['messages'][-1]['content']
def tokens(chunks):
    return tuple(sum((c.usage_metadata or {}).get(k,0) for c in chunks) for k in ('input_tokens','output_tokens'))


class AsyncBody(httpx.AsyncByteStream):
    def __init__(self,data,failure=None):self.data,self.failure,self.closed=data,failure,False
    async def __aiter__(self):
        for start in range(0,len(self.data),31):
            await asyncio.sleep(0)  # Deliberately interleave simultaneous calls.
            yield self.data[start:start+31]
        if self.failure:raise self.failure
    async def aclose(self):self.closed=True


def model(**clients):
    return _CumulativeUsageChatOpenAI(model='deepseek-v4-flash',api_key='offline-placeholder',
        base_url='https://api.deepseek.com/v1',max_tokens=10000,streaming=True,stream_usage=True,
        max_retries=0,**clients)


@pytest.mark.parametrize('second',['B','equal'])
async def test_consecutive_higher_and_identical_tail_usage_are_full_per_call(second):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,stream=AsyncBody(body(label(r)))))) as client:
        m=model(http_async_client=client)
        first=[c async for c in m.astream('A')];next_call=[c async for c in m.astream(second)]
    assert tokens(first)==SAMPLES['A'][:2]
    assert tokens(next_call)==SAMPLES[second][:2]
    assert m._prev_usage=={}


async def test_concurrent_async_calls_share_sdk_client_but_not_usage_state():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,stream=AsyncBody(body(label(r)))))) as client:
        m=model(http_async_client=client)
        async def consume(name):return tokens([c async for c in m.astream(name)])
        assert await asyncio.gather(consume('A'),consume('B'))==[SAMPLES['A'][:2],SAMPLES['B'][:2]]
        assert m.http_async_client is client
        assert m.max_tokens==10000 and m.model_name=='deepseek-v4-flash'
    assert m._prev_usage=={}


async def test_two_async_generators_interleaved_in_same_task_are_isolated():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,stream=AsyncBody(body(label(r)))))) as client:
        m=model(http_async_client=client);streams=[m.astream('A'),m.astream('B')]
        chunks=[[],[]];done=set()
        while len(done)<2:
            for index,stream in enumerate(streams):
                if index in done:continue
                try:chunks[index].append(await anext(stream))
                except StopAsyncIteration:done.add(index)
        assert [tokens(c) for c in chunks]==[SAMPLES['A'][:2],SAMPLES['B'][:2]]
    assert m._prev_usage=={}


def test_sync_sequential_and_same_thread_interleaved_generators():
    with httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,content=body(label(r))))) as client:
        m=model(http_client=client)
        assert tokens(list(m.stream('A')))==SAMPLES['A'][:2]
        assert tokens(list(m.stream('B')))==SAMPLES['B'][:2]
        streams=[m.stream('A'),m.stream('B')];chunks=[[],[]];done=set()
        while len(done)<2:
            for index,stream in enumerate(streams):
                if index in done:continue
                try:chunks[index].append(next(stream))
                except StopIteration:done.add(index)
        assert [tokens(c) for c in chunks]==[SAMPLES['A'][:2],SAMPLES['B'][:2]]
    assert m._prev_usage=={}


def test_shared_model_in_two_threads_keeps_usage_independent():
    barrier=threading.Barrier(2)
    def respond(request):
        barrier.wait(timeout=5)
        return httpx.Response(200,content=body(label(request)))
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        m=model(http_client=client)
        with ThreadPoolExecutor(max_workers=2) as workers:
            result=list(workers.map(lambda name:tokens(list(m.stream(name))),['A','B']))
    assert result==[SAMPLES['A'][:2],SAMPLES['B'][:2]]
    assert m._prev_usage=={}


async def test_failed_or_closed_stream_does_not_contaminate_following_call():
    responses=[]
    def respond(request):
        data=body(label(request))
        if label(request)=='A':data=data.replace(b'data: [DONE]\n\n',b'')
        stream=AsyncBody(data,httpx.ReadError('offline') if label(request)=='A' else None)
        responses.append(stream)
        return httpx.Response(200,stream=stream)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        m=model(http_async_client=client)
        with pytest.raises(httpx.ReadError):
            _=[c async for c in m.astream('A')]
        assert tokens([c async for c in m.astream('B')])==SAMPLES['B'][:2]
        stream=m._astream([HumanMessage(content='B')])
        await anext(stream)
        await stream.aclose()
        # SDK/parent-generator resource cleanup timing is outside this change;
        # closing a call must not leave Usage state on the shared model.
        assert m._prev_usage=={}
        assert tokens([c async for c in m.astream('B')])==SAMPLES['B'][:2]
    assert m._prev_usage=={}
