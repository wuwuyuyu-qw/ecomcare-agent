"""Replay saved provider Usage observations, not reconstructed model answers.

Source: recruit_sprint_20261024_123726/baseline12-after-local-recovery.
The HTTP envelopes below are synthetic; the 61 observed usage/status tuples are
copied from the immutable ledger. Rerank response bodies were not retained.
"""
from decimal import Decimal
import json
import sqlite3

import httpx
import pytest

from app.evaluation import dev_batch as batch, money_guard as money
from app.evaluation.stage1 import ProviderBudgetExceeded

SOURCE_LEDGER_SHA256 = "3bfe2b8bf2bbbf2caf2e4e77080ee3b73b9f1bed3d01f7d2e15ed2400fd3ba9d"
RECORDED = [
    (1, 'chat', 200, 'deepseek-flash', 382, 29, 128),
    (2, 'chat', 200, 'deepseek-flash', 874, 50, 640),
    (3, 'chat', 200, 'deepseek-flash', 615, 117, 384),
    (4, 'chat', 200, 'deepseek-flash', 1380, 46, 128),
    (5, 'chat', 200, 'deepseek-flash', 383, 28, 128),
    (6, 'chat', 200, 'deepseek-flash', 875, 50, 640),
    (7, 'chat', 200, 'deepseek-flash', 616, 121, 384),
    (8, 'chat', 200, 'deepseek-flash', 1263, 47, 128),
    (9, 'chat', 200, 'deepseek-flash', 386, 32, 252),
    (10, 'chat', 200, 'deepseek-flash', 878, 50, 640),
    (11, 'chat', 200, 'deepseek-flash', 619, 121, 384),
    (12, 'chat', 200, 'deepseek-flash', 440, 82, 256),
    (13, 'retrieval', 200, 'BAAI/bge-m3', 10, 0, None),
    (14, 'retrieval', 200, None, None, None, None),
    (15, 'retrieval', 200, 'BAAI/bge-m3', 5, 0, None),
    (16, 'retrieval', 200, None, None, None, None),
    (17, 'chat', 200, 'deepseek-flash', 1364, 49, 128),
    (18, 'chat', 200, 'deepseek-flash', 387, 37, 128),
    (19, 'chat', 200, 'deepseek-flash', 879, 51, 640),
    (20, 'chat', 200, 'deepseek-flash', 879, 51, 640),
    (21, 'chat', 200, 'deepseek-flash', 620, 123, 384),
    (22, 'chat', 200, 'deepseek-flash', 441, 83, 256),
    (23, 'retrieval', 200, 'BAAI/bge-m3', 11, 0, None),
    (24, 'retrieval', 200, None, None, None, None),
    (25, 'retrieval', 200, 'BAAI/bge-m3', 5, 0, None),
    (26, 'retrieval', 200, None, None, None, None),
    (27, 'chat', 200, 'deepseek-flash', 1255, 50, 128),
    (28, 'chat', 200, 'deepseek-flash', 385, 62, 128),
    (29, 'chat', 200, 'deepseek-flash', 878, 58, 640),
    (30, 'chat', 200, 'deepseek-flash', 878, 50, 640),
    (31, 'chat', 200, 'deepseek-flash', 618, 123, 384),
    (32, 'chat', 200, 'deepseek-flash', 1276, 48, 128),
    (33, 'chat', 200, 'deepseek-flash', 384, 78, 128),
    (34, 'chat', 200, 'deepseek-flash', 879, 50, 640),
    (35, 'chat', 200, 'deepseek-flash', 617, 121, 384),
    (36, 'chat', 200, 'deepseek-flash', 1259, 47, 128),
    (37, 'chat', 200, 'deepseek-flash', 385, 83, 251),
    (38, 'chat', 200, 'deepseek-flash', 877, 51, 640),
    (39, 'chat', 200, 'deepseek-flash', 877, 51, 640),
    (40, 'chat', 200, 'deepseek-flash', 618, 120, 384),
    (41, 'chat', 200, 'deepseek-flash', 385, 85, 128),
    (42, 'chat', 200, 'deepseek-flash', 877, 51, 640),
    (43, 'chat', 200, 'deepseek-flash', 877, 51, 640),
    (44, 'chat', 200, 'deepseek-flash', 618, 120, 384),
    (45, 'chat', 200, 'deepseek-flash', 386, 232, 128),
    (46, 'chat', 200, 'deepseek-flash', 875, 50, 640),
    (47, 'chat', 200, 'deepseek-flash', 619, 120, 384),
    (48, 'chat', 200, 'deepseek-flash', 386, 142, 128),
    (49, 'chat', 200, 'deepseek-flash', 878, 50, 640),
    (50, 'chat', 200, 'deepseek-flash', 619, 120, 384),
    (51, 'chat', 200, 'deepseek-flash', 378, 45, 128),
    (52, 'chat', 200, 'deepseek-flash', 870, 50, 640),
    (53, 'chat', 200, 'deepseek-flash', 611, 99, 384),
    (54, 'chat', 200, 'deepseek-flash', 384, 43, 128),
    (55, 'chat', 200, 'deepseek-flash', 876, 50, 640),
    (56, 'chat', 200, 'deepseek-flash', 617, 120, 384),
    (57, 'chat', 200, 'deepseek-flash', 1267, 47, 128),
    (58, 'chat', 200, 'deepseek-flash', 403, 146, 256),
    (59, 'chat', 200, 'deepseek-flash', 886, 50, 640),
    (60, 'chat', 200, 'deepseek-flash', 624, 129, 384),
    (61, 'chat', 200, 'deepseek-flash', 1288, 57, 128),
]


def replay_saved_sequence(tmp_path, monkeypatch):
    path=tmp_path/'replay.sqlite';batch.initialize(path)
    monkeypatch.setattr(money,'query_balance',lambda key:Decimal('15.31'))
    money.activate(path,'offline-placeholder')
    monkeypatch.setattr(httpx.Client,'send',httpx.Client.send)
    monkeypatch.setattr(httpx.AsyncClient,'send',httpx.AsyncClient.send)
    batch.install_http_budget(path,{'api.deepseek.com','api.siliconflow.cn'})
    sent=[];blocked=[]
    def transport(request):
        row=RECORDED[len(sent)];sent.append(row[0])
        _,kind,status,model,incoming,outgoing,cache=row
        if incoming is None:return httpx.Response(status,json={})
        usage={'prompt_tokens':incoming,'completion_tokens':outgoing,'total_tokens':incoming+outgoing}
        if cache is not None:usage['prompt_cache_hit_tokens']=cache
        return httpx.Response(status,json={'model':model,'usage':usage})
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        for attempt,kind,status,model,incoming,outgoing,cache in RECORDED:
            if kind=='chat':
                url='https://api.deepseek.com/v1/chat/completions'
                body={'model':'deepseek-v4-flash','max_tokens':10000}
            else:
                endpoint='rerank' if incoming is None else 'embeddings'
                url='https://api.siliconflow.cn/v1/'+endpoint
                body={'model':'BAAI/bge-reranker-v2-m3' if endpoint=='rerank' else 'BAAI/bge-m3'}
            try:client.post(url,json=body)
            except ProviderBudgetExceeded:blocked.append(attempt)
    state=money.snapshot(path)
    with sqlite3.connect(path) as conn:
        missing=conn.execute('SELECT COUNT(*) FROM provider_usage WHERE attempt_id=14').fetchone()[0]
    return {'recording_sha256':SOURCE_LEDGER_SHA256,'recorded_requests':len(RECORDED),
        'mock_transport_attempt_ids':sent,'blocked_original_attempt_ids':blocked,
        'first_missing_usage_attempt_id':14,'original_first_missing_case':'task-02-01',
        'missing_usage_was_not_fabricated':missing==0,'state':state,
        'real_provider_requests':0,'real_wallet_requests':0,'original_ledger_modified':False,
        'scope':'saved usage/status observations replayed in synthetic JSON HTTP envelopes, not byte-identical provider bodies'}


def test_recorded_first_rerank_usage_gap_blocks_all_47_following_requests(tmp_path,monkeypatch):
    result=replay_saved_sequence(tmp_path,monkeypatch)
    assert result['mock_transport_attempt_ids']==list(range(1,15))
    assert result['blocked_original_attempt_ids']==list(range(15,62))
    assert result['missing_usage_was_not_fabricated']
    state=result['state']
    assert state['provider_usage']['retrieval']['input_tokens'] is None
    assert state['provider_usage']['retrieval']['usage_unknown_requests']==1
    assert state['unknown_usage_requests']==1
    assert state['free_retrieval_estimated_cost_cny']=='0'
    assert state['unknown_cost_requests']==0  # Unknown free Token != unknown paid bill.
    assert 'retrieval_usage_unavailable_or_unreliable' in state['blocks']


def test_inflight_free_retrieval_blocks_other_provider_admission(tmp_path,monkeypatch):
    path=tmp_path/'inflight.sqlite';batch.initialize(path)
    monkeypatch.setattr(money,'query_balance',lambda key:Decimal('15.31'))
    money.activate(path,'offline-placeholder')
    batch.reserve(path,'retrieval',0,20)
    with pytest.raises(ProviderBudgetExceeded,match='retrieval_usage_unresolved'):
        batch.reserve(path,'chat',10000,20)


@pytest.mark.parametrize('payload',[
    {'usage':{'prompt_tokens':10,'completion_tokens':0,'total_tokens':10}},
    {'usage':{'prompt_tokens':10,'completion_tokens':0,'total_tokens':999}},
])
def test_valid_free_usage_can_continue_and_conflicting_usage_cannot(tmp_path,monkeypatch,payload):
    path=tmp_path/'free.sqlite';batch.initialize(path)
    monkeypatch.setattr(money,'query_balance',lambda key:Decimal('15.31'))
    money.activate(path,'offline-placeholder')
    monkeypatch.setattr(httpx.Client,'send',httpx.Client.send)
    monkeypatch.setattr(httpx.AsyncClient,'send',httpx.AsyncClient.send)
    batch.install_http_budget(path,{'api.siliconflow.cn','api.deepseek.com'})
    with httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,json=payload))) as client:
        client.post('https://api.siliconflow.cn/v1/rerank',json={'model':'BAAI/bge-reranker-v2-m3'})
    if payload['usage']['total_tokens']==10:
        batch.reserve(path,'chat',10000,20)
    else:
        with pytest.raises(ProviderBudgetExceeded):batch.reserve(path,'chat',10000,20)
