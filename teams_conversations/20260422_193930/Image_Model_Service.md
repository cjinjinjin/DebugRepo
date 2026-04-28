# Teams Messages: Image Model Service
Period: 2026-04-15 to 2026-04-22
Messages: 107


### 2026-04-22

**Qingjun Tian** (2026-04-22 07:20)
  你打log的code好像不全吧：

**Qingjun Tian** (2026-04-22 07:11)
  call一下把

**Junyi Shen** (2026-04-22 07:10)
  Qingjun Tian 你能把打log的codeshare一下吗？ model.py - Repos

**Qingjun Tian** (2026-04-22 07:06)
  siwen是不是更新了cert？

**Qingjun Tian** (2026-04-22 07:03)
  你能把打log的codeshare一下吗？

**Qingjun Tian** (2026-04-22 07:03)
  我看到relevence的log已经有了

**Junyi Shen** (2026-04-22 07:00)
  Qingjun Tian 这个是今天发的 request？ 是的

**Junyi Shen** (2026-04-22 07:00)
  Siwen Zhu 好的，现在的代码里都kustolog都还是默认写到prod环境的，我先deploy上，晚点代码改到si环境，再替换上。 会不会是和siwen一样的问题？我之前是能在SI查到部分'ImgDiversityModel'的log，但今天的trackingid又查不到了

**Qingjun Tian** (2026-04-22 06:57)
  这个是今天发的 request？

**Qingjun Tian** (2026-04-22 06:56)
  所以你确定output了log了吗？

**Junyi Shen** (2026-04-22 06:56)
  appsvc_info | union appsvc_warn | union appsvc_err | where Timestamp &gt; ago(1d) | where TrackingId == '64903012-aa00-4d19-9aa1-be353dad4417' 这个trackingid能查到 “Fatal model failure(s): Image Diversity base64 request failed.” 但筛选diversity的log后是没有结果的 | where ApplicationName == 'ImgDiversityModel' &nbsp;

**Qingjun Tian** (2026-04-22 06:53)
  确定没有进吗？

**Junyi Shen** (2026-04-22 06:53)
  如果还没有进入model eval，这个怎么查原因呢

**Qingjun Tian** (2026-04-22 06:53)
  AIGC这边的log只能看到调用model fail了，没有返回 - 大概率是超时了。

**Qingjun Tian** (2026-04-22 06:52)
  Junyi Shen 我测试并发量&lt;3时，失败率很低；并发量=5，还是有2%左右的fail request。查trackingid的报错是Fatal model failure(s): Image Diversity base64 request failed. Qingjun Tian 有没有其他log可以看到这个failure的原因，是否是diversity model eval的问题呢 这个要看你的log了呀

**Qingjun Tian** (2026-04-22 06:51)
  你每次有deployment，不应该改这个名字

**Qingjun Tian** (2026-04-22 06:51)
  我们最好能有一个descriptive的名字，而且是轻易不变的

**Junyi Shen** (2026-04-22 06:51)
  我改下命名

**Qingjun Tian** (2026-04-22 06:51)
  给你看一些别的service的，

**Qingjun Tian** (2026-04-22 06:50)
  &nbsp; &quot;route/AdsUnified.MistralCopilotForSI&quot; &nbsp;&quot;route/AdsUnified.AutoImageEditSI&quot; &quot;route/AdsUnified.AutoPromptRewriteSI&quot; &nbsp;

**Qingjun Tian** (2026-04-22 06:49)
  以后都是这样吗？

**Qingjun Tian** (2026-04-22 06:49)
  route/PicassoAdsCreative.ImageDiversityModelTestA100MIG7V2

**Qingjun Tian** (2026-04-22 06:49)
  @mentions: Junyi Shen
  Junyi Shen , 这个SI 的endpoint

**Junyi Shen** (2026-04-22 06:46)
  @mentions: Qingjun Tian
  Qingjun Tian Diversity这个model，enable之前，咱们这边还有什么已知的问题吗？ 我测试并发量&lt;3时，失败率很低；并发量=5，还是有2%左右的fail request。查trackingid的报错是Fatal model failure(s): Image Diversity base64 request failed. &nbsp; Qingjun Tian 有没有其他log可以看到这个failure的原因，是否是diversity model eval的问题呢

**Junyi Shen** (2026-04-22 06:43)
  ImageModelPerfPred.xlsx Qingjun Tian 更新完了ping我一下 ImageModelPerfPred.xlsx 已经更新好了

**Judy Wu (STC)** (2026-04-22 06:40)
  应该做成配置

**Judy Wu (STC)** (2026-04-22 06:40)
  不是，SI 写到SI环境，Prod写到Prod环境

**Siwen Zhu** (2026-04-22 06:36)
  Judy Wu (STC) A100_Train现在有6台，除去给outpainting, T2I做测试用的，可以分2台做relevance SI. Siwen Zhu 你看一下deploy 2台机器做relevance SI？ 好的，现在的代码里都kustolog都还是默认写到prod环境的，我先deploy上，晚点代码改到si环境，再替换上。

**Qingjun Tian** (2026-04-22 06:36)
  Diversity这个model，enable之前，咱们这边还有什么已知的问题吗？

**Qingjun Tian** (2026-04-22 06:34)
  更新完了ping我一下

**Junyi Shen** (2026-04-22 06:33)
  Judy Wu (STC) diversity 需要Mig7机器应该足够, Junyi Shen 你check一下能deploy 2台机器做SI么？ 2台mig 7是有的，我deploy好后更新到endpoint文档里

**Qingjun Tian** (2026-04-22 06:32)
  早晨开会，text relevance缺SI 环境被说成是pilot blocker了，我们也抓紧补上吧，特别是relevance的

**Judy Wu (STC)** (2026-04-22 06:32)
  @mentions: Siwen Zhu
  A100_Train现在有6台，除去给outpainting, T2I做测试用的，可以分2台做relevance SI.&nbsp; Siwen Zhu &nbsp;你看一下deploy 2台机器做relevance SI？

**Judy Wu (STC)** (2026-04-22 06:30)
  @mentions: Junyi Shen
  diversity 需要Mig7机器应该足够, Junyi Shen &nbsp;你check一下能deploy 2台机器做SI么？

**Qingjun Tian** (2026-04-22 06:20)
  咱们的ImageRelevance/Diversity model， 估计什么时候能有可用的SI model？


### 2026-04-21

**Siwen Zhu** (2026-04-21 10:00)
  @mentions: Jinjin
  Siwen Zhu Relevance model的endpoint是这个 https://NorthCentralUS.bing.prod.dlis.binginternal.com/route/PicassoAdsCreative.ImageLPRelevanceSco… 旧的这个 https://WestUS2.bing.prod.dlis.binginternal.com/route/PicassoAdsC… 释放了&nbsp; Jinjin

**Qingjun Tian** (2026-04-21 09:58)
  没有了

**Siwen Zhu** (2026-04-21 09:56)
  Relevance model的endpoint是这个 https://NorthCentralUS.bing.prod.dlis.binginternal.com/route/PicassoAdsCreative.ImageLPRelevanceSco… 旧的这个 https://WestUS2.bing.prod.dlis.binginternal.com/route/PicassoAdsCreative.ImageLPRelevance 现在还有人在调用吗？如果没有的话，我就删掉释放资源了。

**Qingjun Tian** (2026-04-21 07:38)
  Qingjun Tian 嗯嗯，总结一下，现在有两个问题 1， AIGC没有发正确的Trackingdata到model侧，我提了一个fix PR，今天争取deploy到SI。 我本地测试了一下，现在能发过去了 2， Image relevance Model hard code了Kustor的配置，所以把SI的log写到Prod DB了。这个siwen也需要改一下 for #2, model现在没有SI和Prod之分，所以现在写得是对的。

**Qingjun Tian** (2026-04-21 07:32)
  嗯嗯，总结一下，现在有两个问题 1， AIGC没有发正确的Trackingdata到model侧，我提了一个fix PR，今天争取deploy到SI。 我本地测试了一下，现在能发过去了 2， Image relevance Model hard code了Kustor的配置，所以把SI的log写到Prod DB了。这个siwen也需要改一下

**Qingjun Tian** (2026-04-21 03:24)
  我先看看code把

**Qingjun Tian** (2026-04-21 03:24)
  刚跟siwencall过了

**Judy Wu (STC)** (2026-04-21 03:23)
  @mentions: Qingjun Tian
  Hi Qingjun Tian &nbsp;Ad strength call relevance用的trackingId和requestId，会跟AIGC call AutoImage的requestId/tracking是一样的么？因为AutoImage会去call relevance model, 不知道这会不会导致relevance model那边分不清两个不同的requests


### 2026-04-16

**Desheng Cui** (2026-04-16 04:05)
  嗯嗯

**Judy Wu (STC)** (2026-04-16 03:52)
  但是现在的做法是需要定期update model，我看看Feng那边是否有auto-renew solution

**Desheng Cui** (2026-04-16 03:45)
  diversity/relevance的

**Desheng Cui** (2026-04-16 03:45)
  Judy Wu (STC) 写kusto log这个应该是统一的证书吧？AutoImage都已经更新完了 对对，就是参考hongwei这个要改一下

**Qingjun Tian** (2026-04-16 03:45)
  AIGC端打印的log &nbsp; &nbsp; datetime(2026-04-15&nbsp;22:32:23.7443566)&nbsp;&nbsp;&nbsp;Send&nbsp;request&nbsp;to&nbsp;https://northcentralus.bing.prod.dlis.binginternal.com/routebatch/PicassoAdsCreative.ImageLPRelevanceScore datetime(2026-04-15&nbsp;22:32:33.1231228)&nbsp;&nbsp;&nbsp;Got&nbsp;response&nbsp;code&nbsp;OK&nbsp;from&nbsp;https://northcentralus.bing.prod.dlis.binginternal.com/routebatch/PicassoAdsCreative.ImageLPRelevanceScore &nbsp;

**Qingjun Tian** (2026-04-16 03:44)
  不对，这个在SI也没有，我刚才看错了： &nbsp; &nbsp; appsvc_info | union appsvc_warn | union appsvc_err | where Timestamp &gt; ago(1d) | where ApplicationName == 'ImgLPRelevanceModel' | where TrackingId == 'a545402e-d8e1-4fc4-8548-206e7dc9a535' &nbsp;

**Judy Wu (STC)** (2026-04-16 03:40)
  写kusto log这个应该是统一的证书吧？AutoImage都已经更新完了

**Desheng Cui** (2026-04-16 03:39)
  dlis写kust log的～

**Desheng Cui** (2026-04-16 03:39)
  新的证书是pfx的，和之前的也不太一样～估计还需要点时间～

**Judy Wu (STC)** (2026-04-16 03:39)
  更新哪个证书? AIGC call DLIS 还是DLIS写Kustlog的？

**Desheng Cui** (2026-04-16 03:39)
  现在的证书到4月底的～

**Qingjun Tian** (2026-04-16 03:38)
  下次什么时候更新证书？

**Desheng Cui** (2026-04-16 03:38)
  下次更新证书的时候一起改吧～

**Qingjun Tian** (2026-04-16 03:38)
  要不然不好join

**Qingjun Tian** (2026-04-16 03:38)
  能改一下不

**Qingjun Tian** (2026-04-16 03:37)
  果然

**Desheng Cui** (2026-04-16 03:37)
  si里能查到么

**Desheng Cui** (2026-04-16 03:37)
  kusto应该是写在si的里面了

**Qingjun Tian** (2026-04-16 03:36)
  这个是Prod env的

**Qingjun Tian** (2026-04-16 03:34)
  但是找不到model 侧的log

**Qingjun Tian** (2026-04-16 03:34)
  appsvc_info | union appsvc_warn | union appsvc_err | where Timestamp &gt; ago(1d) | where TrackingId == 'a545402e-d8e1-4fc4-8548-206e7dc9a535' &nbsp; 这个我看应该调用了Image relevance

**Siwen Zhu** (2026-04-16 03:32)
  appsvc_info | union appsvc_warn | union appsvc_err | where Timestamp &gt; ago(1d) | where ApplicationName == 'ImgLPRelevanceModel'

**Qingjun Tian** (2026-04-16 03:30)
  Junyi Shen 这个只是diversity的 知道relevance的怎么查吗？

**Qingjun Tian** (2026-04-16 03:28)
  这个最终是成功了的

**Qingjun Tian** (2026-04-16 03:28)
  &nbsp; getaiservicelog( 'a545402e-d8e1-4fc4-8548-206e7dc9a535' , ago(2d)) &nbsp;

**Junyi Shen** (2026-04-16 03:28)
  Junyi Shen where ApplicationName == 'ImgDiversityModel' 这个只是diversity的

**Junyi Shen** (2026-04-16 03:27)
  Qingjun Tian Junyi Shen， model的log写到哪个kusto了？我应该怎么查？ 之前failed的报错，这几个trackid我查过，还没有进到模型inference，所以appsvr也没有查到这个trackid &quot;Fatal model failure(s): Image Diversity base64 request failed.&quot;&nbsp;

**Qingjun Tian** (2026-04-16 03:26)
  这个没有relevance model的log呀

**Qingjun Tian** (2026-04-16 03:25)
  我就是没搜到呢。。 &nbsp; getaiservicelog( '5cf88841-818b-4e6b-b996-ae113330ca51' , ago(2d))

**Junyi Shen** (2026-04-16 03:25)
  where ApplicationName == 'ImgDiversityModel'

**Qingjun Tian** (2026-04-16 03:24)
  @mentions: Junyi Shen
  Junyi Shen ， model的log写到哪个kusto了？我应该怎么查？

**Siwen Zhu** (2026-04-16 01:20)
  ok

**Judy Wu (STC)** (2026-04-16 01:19)
  @mentions: Siwen Zhu
  Siwen Zhu &nbsp;could you please investigate and check Kusto log what happened?

**Qingjun Tian** (2026-04-16 01:03)
  查了一下prod上，image relevance的latency有时候还是挺高的

**Qingjun Tian** (2026-04-16 01:03)
  Execute: [ Web ] [ Desktop ] [ Web (Lens) ] [ Desktop (SAW) ] https://bingads.kusto.windows.net/BingAdsTracing GetPerformancePredictionScore_Stas(ago(12h), now()) | project TrackingId , Timestamp , imgCount , ImageRelevanceModel_dur &nbsp; &nbsp; TrackingId Timestamp imgCount ImageRelevanceModel_dur 5cf88841-818b-4e6b-b996-ae113330ca51 2026-04-15 19:41:00.8974662 5 10012 560709e6-143c-4e9f-a69d-1e18ae821356 2026-04-15 17:28:27.1683382 3 9730 a545402e-d8e1-4fc4-8548-206e7dc9a535 2026-04-15 22:32:23.4119073 3 9380 769ce376-3953-4088-b000-4032ca73856d 2026-04-15 17:28:28.8367807 3 7250 18d8a6f2-0fb6-42d7-ac90-9b9d326ca9a3 2026-04-15 19:41:07.7559436 6 6589 e64be024-afab-4796-a9ea-b865656e488f 2026-04-15 23:04:28.0726844 6 1884 e1dee09e-6747-4e1f-bf28-0ad9681bed4a 2026-04-15 17:30:01.3205381 3 470 e2483604-b1a9-4285-827e-81f0d6cc907d 2026-04-15 19:50:04.7037349 5 464 0e6e25f5-dfb1-423f-a862-e3de5d34f0a2 2026-04-16 00:50:46.7454090 5 338 829cd030-8ce4-44f7-b6d0-305d91c59142 2026-04-15 17:29:59.8761340 2 294 af213df0-b736-44aa-82a4-763732ac8392 2026-04-15 17:29:58.0460272 1 245 052273a8-b290-40e8-b0d1-ed97a9f78468 2026-04-15 17:30:04.1449193 4 225 6d0f5d0d-00bc-4d0a-9126-c35df478cc7c 2026-04-15 17:29:54.5386895 1 0 f4b8a41a-5802-4589-a622-716a830f10ed 2026-04-15 23:04:36.3027790 4 0 85abb719-fe56-4022-bcc0-d140a17df516 2026-04-15 22:32:22.7425378 0 0 e2986d74-f824-4c8c-a1b9-8962aeadd40c 2026-04-15 23:04:34.6137496 5 0 &nbsp;


### 2026-04-15

**Judy Wu (STC)** (2026-04-15 09:04)
  这种request的QPS应该特别小吧?尤其是那些A100 serve的model本来支持的QPS就有限

**Judy Wu (STC)** (2026-04-15 09:04)
  Got it, 你是担心SI打到Prod影响了Prod？

**Judy Wu (STC)** (2026-04-15 09:03)
  凡是上到Prod的model，new version一定得有SI环境去测试之后，才能替换Prod

**Qingjun Tian** (2026-04-15 09:03)
  打到AIGC的SI，然后到了model的prod，我理解这样有可能影响model的capacity

**Qingjun Tian** (2026-04-15 09:03)
  也有打到SI的，

**Judy Wu (STC)** (2026-04-15 09:02)
  那这跟分SI和Prod有关联么？periodic monitoring一定是要打到Prod上的

**Qingjun Tian** (2026-04-15 09:01)
  是的

**Judy Wu (STC)** (2026-04-15 09:01)
  periodic monitoring是说即使没有广告主的流量，AIGC也会发requests去看API availablilty之类的？

**Qingjun Tian** (2026-04-15 09:00)
  AIGC有自己的periodic monitoring，要真的打流量到aigc的endpoint

**Judy Wu (STC)** (2026-04-15 08:59)
  monitor应该是对着kusto log吧？难道会request prod endpoint?&nbsp;

**Qingjun Tian** (2026-04-15 08:58)
  Judy Wu (STC) text relevance分了SI和Prod? 目前也没有呢。。

**Qingjun Tian** (2026-04-15 08:58)
  总之现在的问题是SI和prod model share同一个endpoint，1）上新feature会比较冒险 2）很多monitor，gated都是直接对着prod endpoint在跑。分走了很多prod的capacity

**Judy Wu (STC)** (2026-04-15 08:57)
  text relevance分了SI和Prod?

**Qingjun Tian** (2026-04-15 08:57)
  Prod

**Judy Wu (STC)** (2026-04-15 08:57)
  backfill现在调的是SI还是Prod?

**Judy Wu (STC)** (2026-04-15 08:56)
  如果要压测的话，2台机器可能不太好测出来效果

**Qingjun Tian** (2026-04-15 08:55)
  SI上的perf我觉得差一点也能接受？

**Judy Wu (STC)** (2026-04-15 08:55)
  diversity可以部署2台作为SI，6台作为Prod

**Qingjun Tian** (2026-04-15 08:54)
  不过也不一定是大量调用导致的。

**Judy Wu (STC)** (2026-04-15 08:54)
  relevance主要是需要A100，这个机器数量比较少。diversity用T4或者Mig7就行，机器相对充裕。

**Qingjun Tian** (2026-04-15 08:54)
  Judy Wu (STC) ok，他们之前测QPS了么？ 据说测了。。。我没详细跟这个，Jason测得

**Qingjun Tian** (2026-04-15 08:53)
  我是说我们自己1）多测一下 2）如果能把SI和prod分开，互相不影响

**Judy Wu (STC)** (2026-04-15 08:53)
  ok，他们之前测QPS了么？

**Qingjun Tian** (2026-04-15 08:53)
  text relevance

**Judy Wu (STC)** (2026-04-15 08:52)
  Qingjun Tian 今天跑backfill，text relevance model非常不稳定， text relevance还是image relevance?&nbsp;

**Junyi Shen** (2026-04-15 08:34)
  Qingjun Tian 有发现吗？ 有可能是 base64_to_pil的问题，另外有两个报错，正在fix

**Qingjun Tian** (2026-04-15 08:20)
  今天跑backfill，text relevance &nbsp;model非常不稳定，

**Qingjun Tian** (2026-04-15 08:20)
  Junyi Shen 我查一下kusto log 有发现吗？

**Qingjun Tian** (2026-04-15 08:19)
  @mentions: Judy Wu (STC)
  Judy Wu (STC) ，我们现在SI和Prod用的同样的model endpoint，估计什么时候能分开？
