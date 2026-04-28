# Teams Messages: Siwen Zhu
Period: 2026-04-15 to 2026-04-22
Messages: 181


### 2026-04-22

**Jinjin Chen** (2026-04-22 09:12)
  &nbsp; namespace用的上prod的，证书用的是si的

**Siwen Zhu** (2026-04-22 09:12)
  啥原因

**Jinjin Chen** (2026-04-22 09:12)
  找到原因了

**Siwen Zhu** (2026-04-22 09:00)
  没有

**Siwen Zhu** (2026-04-22 09:00)
  code里吗

**Jinjin Chen** (2026-04-22 08:59)
  看看你那里有e65e832b-d26e-4d59-be94-d261cd10435c 这个文件吗

**Siwen Zhu** (2026-04-22 08:59)
  我不知道

**Jinjin Chen** (2026-04-22 08:59)
  是有个文件吗

**Jinjin Chen** (2026-04-22 08:58)
  DLIS_SCOPE&nbsp;=&nbsp;os.environ.get( &quot;DLIS_SCOPE&quot; ,&nbsp; &quot;e65e832b-d26e-4d59-be94-d261cd10435c/.default&quot; ) 这个参数是什么啊

**Siwen Zhu** (2026-04-22 08:58)
  你要是一直不好查的，麻烦点可能就本地起服务多打点log看下kusto初始化哪些好了没

**Jinjin Chen** (2026-04-22 08:57)
  哦哦，现在服务已经启动了

**Siwen Zhu** (2026-04-22 08:57)
  我以为你是想查看服务启动的问题

**Siwen Zhu** (2026-04-22 08:56)
  是的&nbsp;

**Jinjin Chen** (2026-04-22 08:56)
  那是Polaris job的log吧，那个我记得没有容器内的print log

**Siwen Zhu** (2026-04-22 08:55)
  哪个不是kusto的也可以看到

**Siwen Zhu** (2026-04-22 08:55)
  我想起来你要看服务启动时候的日志可以看polaris log

**Jinjin Chen** (2026-04-22 08:54)
  看到了

**Jinjin Chen** (2026-04-22 08:54)
  嗯嗯

**Siwen Zhu** (2026-04-22 08:54)
  或者直接搜这个trakingid试试

**Siwen Zhu** (2026-04-22 08:53)
  

**Siwen Zhu** (2026-04-22 08:53)
  看最新的

**Siwen Zhu** (2026-04-22 08:53)
  要按时间排序一下

**Jinjin Chen** (2026-04-22 08:53)
  

**Jinjin Chen** (2026-04-22 08:53)
  但没看到你刚刚测试的那个

**Siwen Zhu** (2026-04-22 08:52)
  你用si证书就是查si

**Jinjin Chen** (2026-04-22 08:52)
  那没问题，可以查到你的

**Siwen Zhu** (2026-04-22 08:52)
  我代码只写到prod了

**Jinjin Chen** (2026-04-22 08:52)
  哦哦

**Siwen Zhu** (2026-04-22 08:52)
  我只有prod

**Jinjin Chen** (2026-04-22 08:52)
  siwen 你是prod si都可以查到吗

**Siwen Zhu** (2026-04-22 08:49)
  appsvc_info | union appsvc_warn | union appsvc_err | where Timestamp &gt;= ago(30min) | where ApplicationName == 'ImgLPRelevanceModel' //'autoimage'// //| where TrackingId == 'aea63068-3294-48bb-9da2-33677787ef92' //| where Message contains &quot;relevance_v2_score&quot; //| where Message contains &quot;Score&quot; or Message contains &quot;threshold&quot; //| where Message contains &quot;'threshold': 0}&quot; //| summarize &nbsp;count()

**Siwen Zhu** (2026-04-22 08:42)
  

**Siwen Zhu** (2026-04-22 06:29)
  

**Siwen Zhu** (2026-04-22 06:28)
  pfx

**Siwen Zhu** (2026-04-22 06:28)
  call_image_lp_relevance.py - Repos

**Siwen Zhu** (2026-04-22 06:28)
  

**Jinjin Chen** (2026-04-22 06:14)
  嗯嗯，延到下周吧

**Siwen Zhu** (2026-04-22 06:08)
  我还没来得及搞

**Siwen Zhu** (2026-04-22 06:08)
  我看这里这个邮件

**Siwen Zhu** (2026-04-22 06:08)
  jinjin周五的会还开吗

**Siwen Zhu** (2026-04-22 06:08)
  

**Jinjin Chen** (2026-04-22 03:58)
  是的

**Siwen Zhu** (2026-04-22 03:56)
  这个就是你的输入吗

**Siwen Zhu** (2026-04-22 03:56)
  

**Jinjin Chen** (2026-04-22 03:33)
  siwen，你昨天说的本地测试也能获取log是指这个吗？ &nbsp; import requests &nbsp; response = requests.post( &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &quot; https://WestUS2.bing.prod.dlis.binginternal.com/route/PicassoAdsCreative.ZImage-V1-Jinjin&quot; , &nbsp;&nbsp;&nbsp; cert=(&quot;/home/jinjinchen/dlis/abo-models/team/dai/auto_image/client/private1.cer&quot;, &quot;/home/jinjinchen/dlis/abo-models/team/dai/auto_image/client/private1.key&quot;), &nbsp;&nbsp;&nbsp; json={&quot;prompt&quot;: &quot;A beautiful sunset over the ocean&quot;, &quot;width&quot;: 1344, &quot;height&quot;: 768}, &nbsp;&nbsp;&nbsp; headers={&quot;Content-Type&quot;: &quot;application/json&quot;}, &nbsp;&nbsp;&nbsp; verify=False, ) print(response.status_code) print(response. Text[:500]) &nbsp; &nbsp; 我试了下确实有输出返回，但看kusto log还是空的：


### 2026-04-21

**Jinjin Chen** (2026-04-21 10:31)
  明白了

**Siwen Zhu** (2026-04-21 10:31)
  是的

**Jinjin Chen** (2026-04-21 10:31)
  哦哦，就是看log的地方不一样

**Siwen Zhu** (2026-04-21 10:31)
  就是si里可以看到

**Siwen Zhu** (2026-04-21 10:31)
  我之前prod环境 配的si的证书

**Siwen Zhu** (2026-04-21 10:31)
  我测试的看起来是证书是什么就是什么

**Jinjin Chen** (2026-04-21 10:30)
  test 也可以吗

**Siwen Zhu** (2026-04-21 10:30)
  这个好像不影响

**Jinjin Chen** (2026-04-21 10:30)
  我看大家配的都是prod的文件

**Jinjin Chen** (2026-04-21 10:29)
  kusto log 是只能提交了prod dlis才能看到吗

**Jinjin Chen** (2026-04-21 10:24)
  估计有点bug

**Jinjin Chen** (2026-04-21 10:23)
  哦哦哦

**Siwen Zhu** (2026-04-21 10:23)
  时间应该不会要两个小时

**Siwen Zhu** (2026-04-21 10:23)
  我一般提了就干别的去了

**Siwen Zhu** (2026-04-21 10:23)
  我之前半小时以内吧

**Siwen Zhu** (2026-04-21 10:23)
  看到了 那我之前真没注意这个

**Siwen Zhu** (2026-04-21 10:22)
  哦哦

**Jinjin Chen** (2026-04-21 10:22)
  不是INstance activate：100%

**Jinjin Chen** (2026-04-21 10:22)
  不是，我是INstance loading： 100%

**Siwen Zhu** (2026-04-21 10:22)
  你这不是actovate吗

**Siwen Zhu** (2026-04-21 10:22)
  

**Jinjin Chen** (2026-04-21 10:21)
  我看别的都是

**Jinjin Chen** (2026-04-21 10:21)
  不是要INstance activate才是成功吗

**Siwen Zhu** (2026-04-21 10:20)
  你这个success就是成功了

**Jinjin Chen** (2026-04-21 10:19)
  &nbsp; siwen这里一般要多久才可以loading完成啊，我这都两个多小时了，还是这个状态

**Jinjin Chen** (2026-04-21 10:11)
  嗯嗯

**Siwen Zhu** (2026-04-21 10:07)
  要不你发给 claudecode 它可以帮你转为具体代码

**Siwen Zhu** (2026-04-21 10:07)
  我已经给也是别人发我的图 一时半会找不到对应的code在哪个机器了

**Jinjin Chen** (2026-04-21 10:01)
  Siwen Zhu siwen 这里的代码可以粘贴给我一下吗

**Jinjin Chen** (2026-04-21 07:21)
  

**Jinjin Chen** (2026-04-21 07:21)
  a100 现在用超了，不知道是不是会自动给下一个job

**Siwen Zhu** (2026-04-21 07:20)
  你看下

**Siwen Zhu** (2026-04-21 07:20)
  哦哦好的 我不知道我删的是a100 还是a100train了

**Jinjin Chen** (2026-04-21 07:19)
  我刚试了下a100 train 是还有，但是那个instance没有cpu了

**Siwen Zhu** (2026-04-21 07:18)
  和a100一样

**Siwen Zhu** (2026-04-21 07:18)
  你可以看看a100train&nbsp;

**Siwen Zhu** (2026-04-21 07:18)
  我删了一个

**Jinjin Chen** (2026-04-21 07:18)
  对的

**Siwen Zhu** (2026-04-21 07:18)
  吗

**Siwen Zhu** (2026-04-21 07:18)
  你是要a100

**Jinjin Chen** (2026-04-21 07:17)
  我看没有空闲的机器了

**Jinjin Chen** (2026-04-21 07:17)
  siwen 你这里有不用的吗

**Siwen Zhu** (2026-04-21 07:01)
  *:Certificate://Thumbprint/02AAAAA5AD5298502C5DC918B2853BA8A8A62FF1,*:AAD://appid/dda2a640-fd3a-4d00-937f-29244b63536e,*:AAD://appid/fa4a9196-4774-46b6-93b7-3e478824b3da,*:AAD://appid/4fb0213c-e983-4210-b494-7b73d650e331,*:Certificate://Thumbprint/eaabe14c574cd0b3ea7f088858633e04fa79d5ad,*:Certificate://Microsoft/dlis.si.advisoraggregator.trafficmanager.net,*:Certificate://Microsoft/dlis.advisoraggregator.trafficmanager.net,*:Certificate://Microsoft/dlis.ci.advisoraggregator.trafficmanager.net,*:Certificate://Thumbprint/98ABB82D13612573438EAA1CBAF9EDE723F4F391

**Jinjin Chen** (2026-04-21 07:01)
  Hi siwen，那个ACL 证书可以发我一下吗


### 2026-04-20

**Jinjin Chen** (2026-04-20 07:33)
  刚发现是log 贴错了，没事了

**Siwen Zhu** (2026-04-20 07:23)
  你说的config是指的什么

**Siwen Zhu** (2026-04-20 07:19)
  没有

**Jinjin Chen** (2026-04-20 07:17)
  siwen 你有遇到过log里用的docker和config里面配置的不是同一个的问题吗？

**Jinjin Chen** (2026-04-20 05:09)
  嗯嗯

**Siwen Zhu** (2026-04-20 04:05)
  quality最近没弄了

**Siwen Zhu** (2026-04-20 04:05)
  我现在弄得是relevancemodel

**Jinjin Chen** (2026-04-20 04:05)
  你后面的qualitymodel也用的这个吗

**Siwen Zhu** (2026-04-20 04:04)
  我是基于之前chang的，他那里用的这个

**Jinjin Chen** (2026-04-20 04:02)
  另外想问一下你，看你用的是vllm.LLM，不是 OaasWrapper，是之前测试的时候不好用吗？还是什么原因呢

**Jinjin Chen** (2026-04-20 03:58)
  嗯嗯，找不到证书就先跳过了，用console输出

**Siwen Zhu** (2026-04-20 03:58)
  不用是代码里没有加证书是把

**Jinjin Chen** (2026-04-20 03:58)
  现在不用证书也没起起来呢

**Siwen Zhu** (2026-04-20 03:57)
  哦哦那你离线怎么测试的

**Jinjin Chen** (2026-04-20 03:57)
  还没走到那一步

**Siwen Zhu** (2026-04-20 03:57)
  证书目录不对起不起来

**Siwen Zhu** (2026-04-20 03:57)
  服务起起来要读证书把

**Jinjin Chen** (2026-04-20 03:57)
  我离线测试的时候还没测过证书，离线是不是测不到这个，因为离线也写不了Kustol?

**Siwen Zhu** (2026-04-20 03:56)
  你测试的时候可以找到吗

**Siwen Zhu** (2026-04-20 03:55)
  都会挂载把

**Jinjin Chen** (2026-04-20 03:55)
  Hi Siwen， DLIS 挂载 Cosmos的时候，只会挂载一级目录吗？我试了下把证书放在一个文件夹里面，就提示找不到了


### 2026-04-17

**Siwen Zhu** (2026-04-17 11:17)
  tools 1.zip

**Siwen Zhu** (2026-04-17 11:14)
  chuntest@br1u43-s2-01:/home/chuntest/ssl_keys 密码：666666

**Siwen Zhu** (2026-04-17 11:13)
  br1t45-s1-01 : /home/chunchen/ssl_keys &nbsp; &nbsp;

**Siwen Zhu** (2026-04-17 11:12)
  这里的private.cert 和private.key

**Siwen Zhu** (2026-04-17 11:12)
  10.224.120.197

**Siwen Zhu** (2026-04-17 11:12)
  /home/siwen/relevance/deploy

**Siwen Zhu** (2026-04-17 11:10)
  Recap: Call with Desheng Cui Thursday, March 26 | Meeting | Microsoft Teams

**Jinjin Chen** (2026-04-17 11:04)
  那还没有，上一个版本报了个错误

**Siwen Zhu** (2026-04-17 11:04)
  那个比较卡时间

**Siwen Zhu** (2026-04-17 11:04)
  还有个bypass得找人手动操作

**Siwen Zhu** (2026-04-17 11:04)
  你是要prod吗

**Jinjin Chen** (2026-04-17 11:04)
  效率+++

**Siwen Zhu** (2026-04-17 11:04)
  牛啊

**Siwen Zhu** (2026-04-17 11:03)
  你就是pr是把

**Jinjin Chen** (2026-04-17 11:03)
  不是就PR build的

**Siwen Zhu** (2026-04-17 11:03)
  不&nbsp;

**Siwen Zhu** (2026-04-17 11:03)
  你是a6000吗

**Jinjin Chen** (2026-04-17 11:03)
  第二次build 更快了，只要8分钟

**Jinjin Chen** (2026-04-17 10:09)
  嗯嗯

**Siwen Zhu** (2026-04-17 10:08)
  config文件里

**Siwen Zhu** (2026-04-17 10:08)
  对

**Jinjin Chen** (2026-04-17 10:07)
  appnam在在哪里设置的呀，代码里面吗

**Siwen Zhu** (2026-04-17 10:06)
  appnam写你自己的

**Siwen Zhu** (2026-04-17 10:06)
  我就用的chun给我的code

**Siwen Zhu** (2026-04-17 10:06)
  appsvc_info | union appsvc_warn | union appsvc_err | where Timestamp &gt; ago(1d) | where ApplicationName == 'ImgLPRelevanceModel'

**Jinjin Chen** (2026-04-17 10:06)
  嗯嗯，这个有没有教程呀，不知道怎么看呀

**Siwen Zhu** (2026-04-17 10:05)
  这个之前的证书还是isi环境的

**Siwen Zhu** (2026-04-17 10:04)
  哦哦 bingadsppe.AdInsightMT | Azure Data Explorer

**Jinjin Chen** (2026-04-17 10:04)
  Siwen Zhu Commit c6283417: fix trackingid bug - Repos 证书 我用的这个

**Siwen Zhu** (2026-04-17 10:03)
  bingads.BingAdsTracing | Azure Data Explorer

**Siwen Zhu** (2026-04-17 10:03)
  你是用的哪个证书

**Jinjin Chen** (2026-04-17 10:03)
  我看chang的文档里到polaris就结束了

**Jinjin Chen** (2026-04-17 10:03)
  kusto log在哪里看呀

**Siwen Zhu** (2026-04-17 09:43)
  

**Jinjin Chen** (2026-04-17 09:42)
  Job - Polaris | ca9f3afd-b362-4f81-b4c3-a87ea56fd217

**Siwen Zhu** (2026-04-17 09:42)
  我第一次试的时候就可以进的

**Siwen Zhu** (2026-04-17 09:42)
  我不知道啊&nbsp;

**Jinjin Chen** (2026-04-17 09:41)
  查看log是要加入SG吗？ 我搜了下没有找到这两个SG

**Siwen Zhu** (2026-04-17 09:38)
  我证书的测了 可以的

**Siwen Zhu** (2026-04-17 09:38)
  okk

**Jinjin Chen** (2026-04-17 09:31)
  Siwen Zhu 我试了下用vllm docker 也能build成功，速度也比较快，半个多小时，离线更快，5分钟以内，不过还没Polaris测试

**Siwen Zhu** (2026-04-17 09:30)
  Job - Polaris | c1b55fd2-51d9-43db-8c23-5f997a376a17

**Jinjin Chen** (2026-04-17 09:30)
  siwen，你那里有最近跑的Polaris job可以分享我一下吗

**Jinjin Chen** (2026-04-17 07:11)
  嗯嗯

**Siwen Zhu** (2026-04-17 07:10)
  这个时新的证书

**Siwen Zhu** (2026-04-17 07:10)
  https://www.cosmos09.osdinfra.net/cosmos/DLISModelRepository/local/xucha/models/ImgLPRelevance6/

**Jinjin Chen** (2026-04-17 07:09)
  siwen 这个文件你那里有吗？可以分享我一下吗 '/Model/AggSvcAuthCert-prod.pfx'

**Siwen Zhu** (2026-04-17 06:55)
  哦哦哦

**Jinjin Chen** (2026-04-17 06:55)
  这个还是老的

**Siwen Zhu** (2026-04-17 06:55)
  你是用的新的dockerbase吗

**Siwen Zhu** (2026-04-17 06:54)
  他修了也过不了可能就是安全问题

**Siwen Zhu** (2026-04-17 06:54)
  我也不知道咋搞

**Siwen Zhu** (2026-04-17 06:54)
  你看看claudecode

**Jinjin Chen** (2026-04-17 06:53)
  还有这个新的问题

**Jinjin Chen** (2026-04-17 06:53)
  emm，我也遇到了，然后用那种方式fix了

**Siwen Zhu** (2026-04-17 06:53)
  我只碰到过pypi 安装源不过security

**Siwen Zhu** (2026-04-17 06:53)
  没有&nbsp;

**Jinjin Chen** (2026-04-17 06:52)
  Docker build 过程中 无法连接到 archive.ubuntu.com 导致 apt-get install 失败，是网络问题 &nbsp; hi siwen 你之前遇到过这个超时的吗

**Siwen Zhu** (2026-04-17 05:30)
  Yes you can ask AI do it. your base image is vllm/vllm-openai:latest You need to install .... pacakage You need to copy which files into the new docke rimage &nbsp; Then AI will generate a docker file and run this docker file, it will build a new docker image &nbsp; You can test this docker image offline and ask AI to save the docker image to a tar and then you can upload the adlsgen2

**Siwen Zhu** (2026-04-17 05:26)
  

**Siwen Zhu** (2026-04-17 05:25)
  Commit c5e4beaa: pypi debug - Repos 没加证书

**Siwen Zhu** (2026-04-17 05:25)
  Commit c6283417: fix trackingid bug - Repos 证书

**Siwen Zhu** (2026-04-17 05:15)
  哦 我看下

**Jinjin Chen** (2026-04-17 05:14)
  你用的是vllm 的模式inference的吗

**Siwen Zhu** (2026-04-17 05:12)
  用耗时太长了 感觉哪里用错了

**Siwen Zhu** (2026-04-17 05:12)
  我不用cot指标特别差

**Jinjin Chen** (2026-04-17 05:07)
  你也可以在你的任务上试试

**Jinjin Chen** (2026-04-17 05:07)
  我的是生成任务，输出不止good,fair这些，不过确实没用thinking模式，对比看下来thinking 模式没有看到明显的增益

**Siwen Zhu** (2026-04-17 04:53)
  jinjin你测的gemma速度是不输出cot的是吧 只输出good/fair/bad

**Jinjin Chen** (2026-04-17 03:15)
  是的

**Siwen Zhu** (2026-04-17 03:03)
  jinjin 你之前vlibration的也是分类任务吗
