const copyButton = document.querySelector('[data-copy-target]');

if (copyButton) {
  copyButton.addEventListener('click', async () => {
    const target = document.getElementById(copyButton.dataset.copyTarget);
    try {
      await navigator.clipboard.writeText(target.innerText);
      copyButton.textContent = 'Copied';
      window.setTimeout(() => { copyButton.textContent = 'Copy'; }, 1600);
    } catch (_) {
      copyButton.textContent = 'Select text';
    }
  });
}
