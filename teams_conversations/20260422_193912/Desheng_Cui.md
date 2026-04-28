# Teams Messages: Desheng Cui
Period: 2026-04-15 to 2026-04-22
Messages: 74


### 2026-04-22

**Jinjin Chen** (2026-04-22 04:07)
  唉

**Desheng Cui** (2026-04-22 04:07)
  嗯嗯

**Jinjin Chen** (2026-04-22 04:07)
  走一遍重新上传？

**Jinjin Chen** (2026-04-22 04:07)
  cosmos还没发rename

**Desheng Cui** (2026-04-22 04:06)
  需要改path

**Jinjin Chen** (2026-04-22 04:06)
  需要改名字吗

**Desheng Cui** (2026-04-22 04:05)
  重新copy？

**Jinjin Chen** (2026-04-22 04:03)
  emm 那咋办

**Desheng Cui** (2026-04-22 04:02)
  gen2上有也未必就不报错

**Desheng Cui** (2026-04-22 04:02)
  https://microsoftapc.sharepoint.com/teams/adsbrain-comm/SitePages/Tools.aspx#adl-data-transfer

**Desheng Cui** (2026-04-22 04:01)
  不过这个好像也有坑

**Jinjin Chen** (2026-04-22 04:01)
  怎么看呀，有路径吗

**Desheng Cui** (2026-04-22 04:01)
  saw上面可以

**Jinjin Chen** (2026-04-22 04:00)
  desheng 你知道可以从哪里检查 Gen2 上的文件吗，我这有个报错说是少了个ckpt 文件，但是看着Gen1-&gt;Gen2 的log 没什么报错


### 2026-04-21

**Desheng Cui** (2026-04-21 09:59)
  嗯嗯

**Jinjin Chen** (2026-04-21 09:59)
  刚找她清理了一下

**Desheng Cui** (2026-04-21 09:44)
  siwen那边也没有么

**Jinjin Chen** (2026-04-21 09:41)
  看着没啥了

**Desheng Cui** (2026-04-21 09:39)
  你只能删点test的

**Jinjin Chen** (2026-04-21 09:39)
  

**Desheng Cui** (2026-04-21 09:39)
  

**Desheng Cui** (2026-04-21 09:39)
  不知道哪里调用的

**Desheng Cui** (2026-04-21 09:38)
  都是prod的

**Desheng Cui** (2026-04-21 09:38)
  这个不能删应该

**Jinjin Chen** (2026-04-21 09:38)
  我看是owner是lixin

**Jinjin Chen** (2026-04-21 09:38)
  这个还需要吗

**Jinjin Chen** (2026-04-21 07:51)
  ok

**Desheng Cui** (2026-04-21 07:49)
  帮忙点个pr？ https://dev.azure.com/msasg/Bing_Ads/_git/RichAdsPipelines/pullrequest/6615195

**Desheng Cui** (2026-04-21 07:27)
  enen

**Jinjin Chen** (2026-04-21 07:26)
  嗯嗯，明天她来了再问问吧

**Desheng Cui** (2026-04-21 07:25)
  那个得问chun～

**Desheng Cui** (2026-04-21 07:25)
  不知道啊

**Jinjin Chen** (2026-04-21 07:25)
  piq 那些还有用吗

**Desheng Cui** (2026-04-21 07:24)
  可以问下siwen，我看应该是有可以删的

**Desheng Cui** (2026-04-21 07:24)
  A100?

**Jinjin Chen** (2026-04-21 07:23)
  我看没有空闲机器了

**Jinjin Chen** (2026-04-21 07:23)
  这里面有不用的job吗


### 2026-04-20

**Desheng Cui** (2026-04-20 07:24)
  有其他人的job在跑把

**Jinjin Chen** (2026-04-20 07:19)
  都不知道为啥 job都fail了，你刷新依然还有新的log

**Desheng Cui** (2026-04-20 07:19)
  应该只run一个job

**Jinjin Chen** (2026-04-20 07:18)
  没问题

**Desheng Cui** (2026-04-20 07:18)
  我理解一个机器那个时间段

**Desheng Cui** (2026-04-20 07:18)
  时间段没问题把

**Jinjin Chen** (2026-04-20 07:18)
  desheng， 你有遇到过log里用的docker和config里面配置的不是同一个的问题吗？ &nbsp; 上午我提到的看log的方式和你提到的方式都是通过筛选机器得到的，我看了下如果机器配置不变的话，一直都是同一台机器，log就全都混在一起了

**Jinjin Chen** (2026-04-20 05:09)
  oo&nbsp;

**Desheng Cui** (2026-04-20 04:12)
  不太清楚～

**Desheng Cui** (2026-04-20 04:11)
  这个应该是chang之前弄的

**Jinjin Chen** (2026-04-20 04:03)
  我看siwen用的是 完全绕过了 OaasWrapper ，直接用 vllm.LLM 初始化 &nbsp; 这个有什么历史原因吗

**Jinjin Chen** (2026-04-20 03:39)
  SELECT &nbsp;&nbsp;&nbsp; machine_name, &nbsp;&nbsp;&nbsp; log_level, &nbsp;&nbsp;&nbsp; log_time, &nbsp;&nbsp;&nbsp; description, &nbsp;&nbsp;&nbsp; pid, &nbsp;&nbsp;&nbsp; tid, &nbsp;&nbsp;&nbsp; title, &nbsp;&nbsp;&nbsp; component, &nbsp;&nbsp;&nbsp; trace_id, &nbsp;&nbsp;&nbsp; file_name, &nbsp;&nbsp;&nbsp; log_offset, &nbsp;&nbsp;&nbsp; environment, &nbsp;&nbsp;&nbsp; machine_function, &nbsp;&nbsp;&nbsp; year, &nbsp;&nbsp;&nbsp; month, &nbsp;&nbsp;&nbsp; day FROM dlissensitivelog&nbsp; WHERE&nbsp; &nbsp; file_name LIKE 'DLMSUserLog_ContainerOutput%.log' &nbsp; -- WHERE file_name LIKE 'TelemetryServerAutoscaleOnboarding%.log' &nbsp; -- AND day = TIMESTAMP '2025-12-02' &nbsp; -- AND log_time BETWEEN TIMESTAMP '2025-12-30 00:00:00' AND TIMESTAMP '2025-12-30 18:00:00' &nbsp; -- AND&nbsp; &nbsp; AND log_time BETWEEN TIMESTAMP '2026-02-27 18:20:00' AND TIMESTAMP '2026-04-28 19:00:00' &nbsp; -- day BETWEEN TIMESTAMP '2025-08-17' AND TIMESTAMP '2025-08-19' &nbsp; -- AND description LIKE '%06e500f1-5bb8-425c-9475-df7d1be4e07e%' &nbsp; -- AND log_level = 'e' &nbsp; -- AND title = 'GpuMonitor' &nbsp; AND machine_name = 'BN2BEAP00005CC8' &nbsp; -- AND environment LIKE 'IndexServeDLModelServe-Prod-MWHE01' LIMIT 10000;

**Desheng Cui** (2026-04-20 03:38)
  SELECT &nbsp;&nbsp;&nbsp; machine_name, &nbsp;&nbsp;&nbsp; log_level, &nbsp;&nbsp;&nbsp; log_time, &nbsp;&nbsp;&nbsp; description, &nbsp;&nbsp;&nbsp; pid, &nbsp;&nbsp;&nbsp; tid, &nbsp;&nbsp;&nbsp; title, &nbsp;&nbsp;&nbsp; component, &nbsp;&nbsp;&nbsp; trace_id, &nbsp;&nbsp;&nbsp; file_name, &nbsp;&nbsp;&nbsp; log_offset, &nbsp;&nbsp;&nbsp; environment, &nbsp;&nbsp;&nbsp; machine_function, &nbsp;&nbsp;&nbsp; year, &nbsp;&nbsp;&nbsp; month, &nbsp;&nbsp;&nbsp; day FROM dlissensitivelog&nbsp; WHERE&nbsp; &nbsp; file_name LIKE 'DLMSUserLog_ContainerOutput%.log' &nbsp; -- WHERE file_name LIKE 'TelemetryServerAutoscaleOnboarding%.log' &nbsp; -- AND day = TIMESTAMP '2025-12-02' &nbsp; -- AND log_time BETWEEN TIMESTAMP '2025-12-30 00:00:00' AND TIMESTAMP '2025-12-30 18:00:00' &nbsp; -- AND&nbsp; &nbsp; AND log_time BETWEEN TIMESTAMP '2026-02-27 18:20:00' AND TIMESTAMP '2026-02-28 19:00:00' &nbsp; -- day BETWEEN TIMESTAMP '2025-08-17' AND TIMESTAMP '2025-08-19' &nbsp; -- AND description LIKE '%06e500f1-5bb8-425c-9475-df7d1be4e07e%' &nbsp; -- AND log_level = 'e' &nbsp; -- AND title = 'GpuMonitor' &nbsp; AND machine_name = 'BN2BEAP0000495E' &nbsp; -- AND environment LIKE 'IndexServeDLModelServe-Prod-MWHE01' LIMIT 10000;

**Desheng Cui** (2026-04-20 03:37)
  https://msasg.visualstudio.com/Bing_and_IPG/_wiki/wikis/Bing_and_IPG.wiki/333909/How-to-Use-Central…

**Desheng Cui** (2026-04-20 03:34)
  users/siwenzhu/VLLM_MML_localbuild_pypi_kusto

**Desheng Cui** (2026-04-20 03:34)
  https://msasg.visualstudio.com/Bing_and_IPG/_git/OaaS_LLMTemplate

**Desheng Cui** (2026-04-20 03:31)
  咋啦

**Desheng Cui** (2026-04-20 03:31)
  enen

**Jinjin Chen** (2026-04-20 03:16)
  desheng，quick call一下？


### 2026-04-15

**Desheng Cui** (2026-04-15 07:47)
  哈哈哈，可以

**Jinjin Chen** (2026-04-15 07:47)
  嗯嗯，把源代码发给cc看看吧

**Desheng Cui** (2026-04-15 07:46)
  主要是vllm我也没玩过~

**Jinjin Chen** (2026-04-15 07:46)
  好吧

**Desheng Cui** (2026-04-15 07:46)
  这个我也不知道。。

**Jinjin Chen** (2026-04-15 07:45)
  postprocess 可以再请求一次吗

**Jinjin Chen** (2026-04-15 07:44)
  

**Desheng Cui** (2026-04-15 07:43)
  上面那个脚本里

**Desheng Cui** (2026-04-15 07:43)
  我记得是batch input

**Jinjin Chen** (2026-04-15 07:41)
  第一步先 生成5个短输出，然后第二步基于第一步的短输出，并行输出5个最终结果

**Desheng Cui** (2026-04-15 07:41)
  还是啥

**Desheng Cui** (2026-04-15 07:40)
  你是想复用cache？

**Jinjin Chen** (2026-04-15 07:40)
  Claude code说现在的设计是只能调用一次

**Jinjin Chen** (2026-04-15 07:39)
  现在的设计是一次只能调用一次vllm吗？我想调用两次，方便第二次生成5个prompt的时候加速

**Desheng Cui** (2026-04-15 07:37)
  qwenvl_inference_with_logprobs.py - Repos

**Jinjin Chen** (2026-04-15 07:36)
  有脚本分享我看看吗

**Desheng Cui** (2026-04-15 07:36)
  用了啊

**Jinjin Chen** (2026-04-15 07:35)
  现在chang和siwen那边在DLIS调用Qwen 用vllm加速了吗
