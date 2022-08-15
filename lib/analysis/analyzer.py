import os
import json
import pandas as pd
from typing import List
from lib.output.verbose import Output
from lib.reports.base import FileBaseReport
from lib.connection.response import Response
from . import identify404


class Analyzer(object):
    def __init__(self, options, output: Output, report: FileBaseReport):
        self.options = options
        self.output = output
        self.report = report
        self.should_save_features = True

    def analysis_responses(self, responses):
        self.output.warning('\nbuild features ...')
        features = self.build_features(responses)

        self.output.warning('\nanalysis features ...')
        labels, results, cluster = identify404.identify_404_by_dbscan(features)

        self.output.warning("\nIdentify404 give cluster information:")
        self.output.warning("\n" + json.dumps(cluster, indent=4))

        existed_responses = []
        for result, response in zip(results, responses):
            if not result:
                continue
            if (200 <= response.status < 300) or (500 <= response.status):
                existed_responses.append(response)

        self.output.warning('\nfound {} existed assets from {} results:'.format(len(existed_responses), len(responses)))
        for response in existed_responses:
            self.output.status_report(response, response.url)

        # 将相关信息记录在日志中
        information = "\nIdentify404 give cluster information:\n"
        information += json.dumps(cluster, indent=4) + "\n"
        information += '\nfound {} existed assets from {} results:\n'.format(len(existed_responses), len(responses))
        information += self.report.generate(existed_responses)
        self.report.save_information(information)

    def build_features(self, responses: List[Response]):
        # 提取原始特征
        original_features = [self.get_response_features(r) for r in responses]

        odf = pd.DataFrame(original_features, columns=identify404.get_404_features_names())
        odf['url'] = [rsp.url for rsp in responses]
        odf['content_type'] = [rsp.type for rsp in responses]
        odf['exists'] = 0

        if self.should_save_features:
            file = os.path.join(os.path.dirname(self.report.output_file), 'features.csv')
            odf.to_csv(file, index=False)

            self.output.warning('\nsave original features to {}'.format(file))

        # 将原始特征预处理成聚类算法所需要的特征
        # 1. 响应内容中的特殊字符计数转化为占比
        rsp_body_count_columns = [name for name in odf.columns if name.startswith('c:')]
        ndf = odf[rsp_body_count_columns].div(odf.standard_body_length, axis=0)
        # 2. 响应状态码转换为哑变量
        ndf = pd.concat([ndf, pd.get_dummies(odf['status_code'], prefix='code')], axis=1)
        # 3. 清理前后响应内容长度变化，并Z-Score标准化
        ndf['body_len_change'] = odf.body_length - odf.standard_body_length
        ndf['body_len_change'] = (ndf.body_len_change - ndf.body_len_change.mean()) / ndf.body_len_change.std()
        # 4. 清理后的响应内容长度Z-Score标准化
        ndf['body_len'] = (odf.standard_body_length - odf.standard_body_length.mean()) / odf.standard_body_length.std()
        # 5. 响应体content-type转换为哑变量
        ndf = pd.concat([ndf, pd.get_dummies(odf['content_type'], prefix='type')], axis=1)

        # ndf = (ndf-ndf.mean())/ndf.std()
        ndf.fillna(0, inplace=True)
        return ndf.to_numpy()

    def get_response_features(self, response: Response):
        url: str = response.url
        status_code: int = response.status
        body: bytes = response.body
        return identify404.get_404_features(url.encode(), status_code, len(body), body)
