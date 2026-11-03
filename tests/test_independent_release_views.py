"""A single multi-view judgment retains both required facts and uncertainty."""
import json

import httpx
import pytest
from test_demo_release_result import release_inputs

from luna_mujoco.demo_perception import DemoVisualInventory
from luna_mujoco.demo_transfer_verification import verify_release_result


@pytest.mark.parametrize('probabilities,status', [
    ((.95,.03), 'at_destination'),
    ((.98,0.), 'at_destination'),
    ((.98,.5), 'unknown'),
    ((.1,0.), 'rejected'),
    ((.98,.99), 'rejected'),
])
def test_multiview_release_acceptance_and_uncertainty(probabilities,status):
    capture,binding,reference=release_inputs()
    for index, view in enumerate(capture['views'].values(), 1):
        view['rgb'] = view['rgb'].copy()
        view['rgb'][:] = index * 50
    calls=[]
    def respond(request):
        payload=json.loads(request.content);calls.append(payload)
        answers=[{'type':'predicate','name':name,'probability':value}
            for name,value in zip(('place_result','workpiece_still_held'),probabilities)]
        return httpx.Response(200,json={'model':'gpt-6-luna','answers':answers})
    inventory=DemoVisualInventory(api_key='test',client=httpx.Client(transport=httpx.MockTransport(respond)))
    result=verify_release_result(inventory,capture,binding,reference)
    visual=result['visual_evidence']
    assert visual['status']==status
    assert len(calls)==1 and visual['camera_ids']==['1','2','3']
    assert 'camera_evidence' not in visual
    assert visual['post_release_metric_pose_measured'] is False
    for payload in calls:
        content=payload['input'][0]['content']
        context=json.loads(content[0]['text'])
        assert len([part for part in content if part['type']=='input_image'])==4
        labels=[part.get('text','') for part in content if part.get('text','').startswith('CURRENT RGB camera')]
        assert labels==['CURRENT RGB camera '+camera for camera in ('1','2','3')]
        assert 'all three CURRENT RGB views together' in context['scope']
        assert 'post-release depth measurement' in context['scope']
        assert 'polygon' not in json.dumps(payload)
