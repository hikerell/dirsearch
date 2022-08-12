import re
import collections
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import urllib
import urllib.parse
import logging

logger = logging.getLogger(__name__)

unimportant_chars = re.compile(rb'\s')
multi_slash = re.compile(rb'//+')
multi_numbers = re.compile(rb'[0-9]+')


def clean_url_from_response_body(request_url: bytes, response_body: bytes):
    if len(request_url) == 0:
        return response_body
    response_body = response_body.replace(request_url, b'')
    try:
        path = urllib.parse.urlparse(request_url).path
        if len(path) > 1:
            # 考虑兼容path为空或者path仅仅包含多个/的场景
            response_body = response_body.replace(path, b'')
        # 考虑兼容url被标准化的情况
        normalized_path = multi_slash.sub(b'/', path)
        if normalized_path != path:
            response_body = response_body.replace(normalized_path, b'')
        # TODO: 考虑windows风格的slash或者../和./的情况
    except:
        logger.exception('failed to clean url: {url} from response')
    response_body = unimportant_chars.sub(b'', response_body)
    return response_body


def clean_numbers_from_response_body(response_body: bytes):
    return multi_numbers.sub(b'0', response_body)


def get_standarized_response_body(request_url: bytes, response_body: bytes):
    request_url = request_url.strip()
    response_body = response_body.strip()
    if len(response_body) == 0:
        return b''
    if len(request_url) > 0:
        response_body = clean_url_from_response_body(request_url, response_body)
    response_body = clean_numbers_from_response_body(response_body)
    return response_body


def get_404_features(request_url: bytes, response_status_code: int, response_body_length: int, response_body: bytes):
    """
    事件404特征:
    响应码，原始响应体长度，标准化响应体长度，多个标准化响应体字符统计特征
    多个标准化响应体字符统计特征包括：<个数，>个数，/个数，{个数，}个数，:个数，"个数，,个数，=个数，(个数，)个数，;个数
    """
    standard_body = get_standarized_response_body(request_url, response_body)
    features = [
        response_status_code, response_body_length, len(standard_body),
        standard_body.count(b'<'), standard_body.count(b'>'), standard_body.count(b'/'),
        standard_body.count(b'</'), standard_body.count(b'/>'), standard_body.count(b'=/'),
        standard_body.count(b'.'), standard_body.count(b"'"),
        standard_body.count(b"["), standard_body.count(b"]"),
        standard_body.count(b"|"), standard_body.count(b"&"),
        standard_body.count(b"+"), standard_body.count(b"-"), standard_body.count(b"*"),
        standard_body.count(b'{'), standard_body.count(b'}'), standard_body.count(b':'),
        standard_body.count(b'"'), standard_body.count(b','), standard_body.count(b'='),
        standard_body.count(b'('), standard_body.count(b')'), standard_body.count(b';')
    ]
    return features


def get_404_features_names():
    names = [
        'status_code', 'body_length', 'standard_body_length',
        'c:<', 'c:>', 'c:/',
        'c:</', 'c:/>', 'c:=/',
        'c:.', "c:'",
        "c:[", "c:]",
        "c:|", "c:&",
        "c:+", "c:-", "c:*",
        'c:{', 'c:}', 'c::',
        'c:"', 'c:,', 'c:=',
        'c:(', 'c:', 'c:;'
    ]
    return names


def identify_404_by_dbscan(data):
    dbscan = DBSCAN()
    labels = dbscan.fit_predict(data)

    clusters = len(set(labels))
    if clusters == 1:
        score = 1
    else:
        score = silhouette_score(data, labels, metric='euclidean')

    counter = collections.Counter(labels)
    label_description = {k: {'count': v, 'ratio': v/len(labels), 'success': False}for k, v in counter.items()}
    sorted_descriptions = sorted(label_description.values(), key=lambda m: m['count'])

    max_ratio = 10 / 100
    current_ratio = 0
    for desc in sorted_descriptions:
        # print(f'current_ratio={current_ratio}', desc)
        if desc['ratio'] + current_ratio > max_ratio:
            break
        desc['success'] = True
        current_ratio += desc['ratio']

    cluster = {
        'bestClusters': clusters,
        'bestScore': score,
        'bestK': len(label_description),
        'labelDescription': None
    }
    results = [label_description[v]['success'] for v in labels]
    cluster['labelDescription'] = {str(k): v for k, v in label_description.items()}
    return labels, results, cluster


def identify_404(events_features, k=5):
    features = np.array(events_features)
    clusters, score, labels = analysis_by_k_means(features, k)
    # final_clusters = len(set())
    counter = collections.Counter(labels)
    label_description = {k: {'count': v, 'ratio': v/len(labels), 'success': False}for k, v in counter.items()}

    sorted_descriptions = sorted(label_description.values(), key=lambda m: m['count'])

    max_ratio = 10 / 100
    current_ratio = 0
    for desc in sorted_descriptions:
        # print(f'current_ratio={current_ratio}', desc)
        if desc['ratio'] + current_ratio > max_ratio:
            break
        desc['success'] = True
        current_ratio += desc['ratio']

    # for desc in label_description.values():
    #     if desc['ratio'] < 0.10:
    #         desc['success'] = True

    cluster = {
        'bestClusters': clusters,
        'bestScore': score,
        'bestK': k,
        'labelDescription': None
    }
    results = [label_description[v]['success'] for v in labels]
    cluster['labelDescription'] = {str(k): v for k, v in label_description.items()}
    return labels, results, cluster


def identify_404_by_search(events_features, max_k=6):
    features = np.array(events_features)
    best_clusters = 0
    best_score = 0.0
    best_labels = []
    best_k = 0
    for k in range(2, max_k):
        k_clusters, k_score, k_labels = analysis_by_k_means(features, k)
        if k_score > best_score:
            best_clusters = k_clusters
            best_score = k_score
            best_labels = k_labels
            best_k = k

    counter = collections.Counter(best_labels)
    label_description = {k: {'label': k, 'count': v, 'ratio': v/len(best_labels), 'success': False}for k, v in counter.items()}

    sorted_descriptions = sorted(label_description.values(), key=lambda m: m['count'])

    print(sorted_descriptions)

    max_ratio = 10 / 100
    current_ratio = 0
    for desc in sorted_descriptions:
        print(f'current_ratio={current_ratio}', desc)
        if desc['ratio'] + current_ratio > max_ratio:
            break
        desc['success'] = True
        current_ratio += desc['ratio']
        print(f'label {1} marked as existed assets')

    # for desc in label_description.values():
    #     if desc['ratio'] < 0.10:
    #         desc['success'] = True

    cluster = {
        'bestClusters': best_clusters,
        'bestScore': best_score,
        'bestK': best_k,
        'labelDescription': None
    }
    results = [label_description[v]['success'] for v in best_labels]
    cluster['labelDescription'] = {str(k): v for k, v in label_description.items()}
    return results, cluster


def analysis_by_k_means(features, k):
    clf = KMeans(n_clusters=k)
    clf.fit(features)
    labels = clf.fit_predict(features)
    clusters = len(set(labels))
    if clusters == 1:
        score = 1
    else:
        score = silhouette_score(features, clf.labels_, metric='euclidean')
    return clusters, score, labels


def identify_404_by_k_means_for_research(events_features, n=8):
    sse = []
    score = []
    events_features = np.array(events_features)
    for k in range(2, n):
        clf = KMeans(n_clusters=k)
        clf.fit(events_features)
        sse.append(clf.inertia_)
        labels = clf.fit_predict(events_features)
        print(f"k={k} final: {len(set(labels))} labels: {labels}")
        score.append(silhouette_score(events_features, clf.labels_, metric='euclidean'))
    return sse, score

