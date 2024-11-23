// 检测（预期功能:提交数据给服务器,接受返回的值，还没做好
document.getElementById('Detect').addEventListener('click', function () {
    chrome.tabs.executeScript({
      code: 'window.alert("此站点为钓鱼网站！")'
    });
  });
  
  // 举报（跳转网信办
  document.getElementById('Report').addEventListener('click', function () {
    chrome.tabs.executeScript({
      code: 'window.location.href = "https://www.12377.cn/wxxx/list1.html"'
    });
  });

  // 退出（关闭网页
  document.getElementById('Quit').addEventListener('click', function () {
    chrome.tabs.executeScript({
      code: 'window.location.href = "https://www.baidu.com"'
    });
  });