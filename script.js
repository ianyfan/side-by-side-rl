function init() {
  const links = [
    ...document.getElementsByTagName("nav")[0].getElementsByTagName("a")
  ].slice(1);
  const linkIndices = Object.fromEntries(
    links.map((element, index) => [element.hash.slice(1), index])
  );

  const observer = new IntersectionObserver(entries => {
    const entry = entries.find(entry => entry.boundingClientRect.bottom > 0);
    if (entry) {
      const current = document.getElementsByClassName("current")[0];
      if (current) current.className = "";

      const link = links[
        linkIndices[entry.target.children[0].id] - !entry.isIntersecting
      ];
      if (link) link.className = "current";
    }
  }, {"rootMargin": "0% 0% -50% 0%"});

  [...document.body.children]
    .filter(element => element.tagName === "ARTICLE")
    .forEach(observer.observe, observer);
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
else init();
