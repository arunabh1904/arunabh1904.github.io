# Auto-add .editor-block to every Rouge output for editor layout
Jekyll::Hooks.register :documents, :post_render do |doc|
  next unless doc.data["layout"] == "editor"
  doc.output.gsub!(
    %r{<div class="highlight">},
    '<div class="highlight editor-block">'
  )
end
